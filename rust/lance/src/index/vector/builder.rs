// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch};
use futures::prelude::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{Error, Result};
use lance_file::v2::{reader::FileReader, writer::FileWriter};
use lance_index::{
    pb,
    vector::{
        ivf::{
            storage::{IvfData, IVF_METADATA_KEY},
            IvfBuildParams,
        },
        quantizer::Quantization,
        transform::Transformer,
        v3::{
            shuffler::{IvfShuffleReader, IvfShuffler},
            storage::{StorageBuilder, VectorStore},
            subindex::IvfSubIndex,
        },
    },
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
};
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, stream::RecordBatchStreamAdapter,
    ReadBatchParams,
};
use lance_linalg::distance::DistanceType;
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};
use tempfile::TempDir;

use crate::Dataset;

use super::{utils, Ivf};

pub struct IvfIndexBuilder<S: IvfSubIndex, Q: Quantization + Clone> {
    dataset: Dataset,
    column: String,
    distance_type: DistanceType,
    shuffler: Box<dyn IvfShuffler>,
    index_dir: String,
    ivf_params: IvfBuildParams,
    sub_index_params: S::BuildParams,
    sub_index: S,
    quantizer: Q,
    temp_dir: Path,
}

impl<S: IvfSubIndex, Q: Quantization + Clone> IvfIndexBuilder<S, Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Dataset,
        column: String,
        distance_type: DistanceType,
        shuffler: Box<dyn IvfShuffler>,
        index_dir: String,
        ivf_params: IvfBuildParams,
        sub_index_params: S::BuildParams,
        sub_index: S,
        quantizer: Q,
    ) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let temp_dir = Path::from(temp_dir.path().to_str().unwrap());
        Ok(Self {
            dataset,
            column,
            distance_type,
            shuffler,
            index_dir,
            ivf_params,
            sub_index_params,
            sub_index,
            quantizer,
            temp_dir,
        })
    }

    pub async fn build(&self) -> Result<()> {
        // step 1. train IVF
        let ivf = self.load_or_build_ivf().await?;

        // step 2. shuffle data
        let reader = self.shuffle_data(ivf.centroids.clone()).await?;
        let partition_build_order = (0..self.ivf_params.num_partitions)
            .map(|partition_id| reader.partiton_size(partition_id))
            .collect::<Result<Vec<_>>>()?
            // sort by partition size in descending order
            .into_iter()
            .enumerate()
            .map(|(idx, x)| (x, idx))
            .sorted()
            .rev()
            .map(|(_, idx)| idx)
            .collect::<Vec<_>>();

        // step 3. build sub index
        let mut partition_sizes = Vec::with_capacity(self.ivf_params.num_partitions);
        for &partition in &partition_build_order {
            let partition_data = reader
                .read_partition(partition)
                .await?
                .ok_or(Error::io(
                    format!("partition {} is empty", partition).as_str(),
                    location!(),
                ))?
                .peekable();
            let batches = partition_data.try_collect::<Vec<_>>().await?;
            let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;

            let sizes = self.build_partition(partition, &batch).await?;
            partition_sizes.push(sizes);
        }

        // step 4. merge all partitions
        self.merge_partitions(partition_sizes).await?;

        Ok(())
    }

    async fn load_or_build_ivf(&self) -> Result<Ivf> {
        let dim = utils::get_vector_dim(&self.dataset, &self.column)?;
        super::build_ivf_model(
            &self.dataset,
            &self.column,
            dim,
            self.distance_type,
            &self.ivf_params,
        )
        .await

        // TODO: load ivf model
    }

    async fn shuffle_data(
        &self,
        centroids: FixedSizeListArray,
    ) -> Result<Box<dyn IvfShuffleReader>> {
        let transformer = Arc::new(lance_index::vector::ivf::new_ivf_with_quantizer(
            centroids,
            self.distance_type,
            &self.column,
            self.quantizer.clone().into(),
            Some(0..self.ivf_params.num_partitions as u32),
        )?);

        let stream = self
            .dataset
            .scan()
            .batch_readahead(num_cpus::get() * 2)
            .project(&[self.column.as_str()])?
            .with_row_id()
            .try_into_stream()
            .await?;

        let mut transformed_stream = Box::pin(
            stream
                .map(move |batch| {
                    let ivf_transformer = transformer.clone();
                    tokio::spawn(async move { ivf_transformer.transform(&batch?) })
                })
                .buffered(num_cpus::get())
                .map(|x| x.unwrap())
                .peekable(),
        );

        let batch = transformed_stream.as_mut().peek().await;
        let schema = match batch {
            Some(Ok(b)) => b.schema(),
            Some(Err(e)) => panic!("do this better: error reading first batch: {:?}", e),
            None => panic!("no data"),
        };

        self.shuffler
            .shuffle(Box::new(RecordBatchStreamAdapter::new(
                schema,
                transformed_stream,
            )))
            .await
    }

    async fn build_partition(&self, part_id: usize, batch: &RecordBatch) -> Result<(usize, usize)> {
        let object_store = ObjectStore::local();

        // build quantized vector storage
        let storage = StorageBuilder::new(
            self.column.clone(),
            self.distance_type,
            self.quantizer.clone(),
        )
        .build(batch)?;
        let path = self.temp_dir.child(format!("storage_part{}", part_id));
        let writer = object_store.create(&path).await?;
        let storage_batch = storage.to_batch()?;
        let mut writer = FileWriter::try_new(
            writer,
            path.to_string(),
            batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&storage_batch).await?;
        let storage_len = writer.finish().await? as usize;

        // build the sub index, with in-memory storage
        self.sub_index
            .index_vectors(&storage, self.sub_index_params.clone())?;

        let path = self.temp_dir.child(format!("index_part{}", part_id));
        let writer = object_store.create(&path).await?;
        let index_batch = self.sub_index.to_batch()?;
        let mut writer = FileWriter::try_new(
            writer,
            path.to_string(),
            index_batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&index_batch).await?;
        let index_len = writer.finish().await? as usize;

        Ok((storage_len, index_len))
    }

    async fn merge_partitions(&self, partition_sizes: Vec<(usize, usize)>) -> Result<()> {
        // prepare the final writers
        let dir_path = Path::parse(self.index_dir.as_str())?;

        let storage_path = dir_path.child(INDEX_AUXILIARY_FILE_NAME);
        let index_path = dir_path.child(INDEX_FILE_NAME);
        // setup the writers later because we prefer to infer the schema from the first batch
        let mut storage_writer = None;
        let mut index_writer = None;

        // maintain the IVF partitions
        let mut storage_ivf = IvfData::empty();
        let mut index_ivf = IvfData::empty();
        let scheduler = ScanScheduler::new(Arc::new(ObjectStore::local()), 64);
        for (part_id, (storage_size, index_size)) in partition_sizes.into_iter().enumerate() {
            if storage_size == 0 {
                storage_ivf.add_partition(0);
                index_ivf.add_partition(0);
                continue;
            }

            let storage_part_path = self.temp_dir.child(format!("storage_part{}", part_id));
            let reader =
                FileReader::try_open(scheduler.open_file(&storage_part_path).await?, None).await?;
            let batch = reader
                .read_stream(ReadBatchParams::RangeFull, storage_size as u32, 1)?
                .try_next()
                .await?
                .ok_or(Error::io("empty storage batch", location!()))?;
            if storage_writer.is_none() {
                storage_writer = Some(FileWriter::try_new(
                    self.dataset.object_store().create(&storage_path).await?,
                    storage_path.to_string(),
                    batch.schema_ref().as_ref().try_into()?,
                    Default::default(),
                )?);
            }
            storage_writer.as_mut().unwrap().write_batch(&batch).await?;

            let index_part_path = self.temp_dir.child(format!("index_part{}", part_id));
            let reader =
                FileReader::try_open(scheduler.open_file(&index_part_path).await?, None).await?;
            let batch = reader
                .read_stream(ReadBatchParams::RangeFull, index_size as u32, 1)?
                .try_next()
                .await?
                .ok_or(Error::io("empty index batch", location!()))?;
            if index_writer.is_none() {
                index_writer = Some(FileWriter::try_new(
                    self.dataset.object_store().create(&index_path).await?,
                    index_path.to_string(),
                    batch.schema_ref().as_ref().try_into()?,
                    Default::default(),
                )?);
            }
            index_writer.as_mut().unwrap().write_batch(&batch).await?;
        }

        let (mut storage_writer, mut index_writer) =
            (storage_writer.unwrap(), index_writer.unwrap());

        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(
            IVF_METADATA_KEY,
            String::from_utf8(storage_ivf_pb.encode_to_vec()).map_err(|e| {
                Error::io(format!("failed to encode IVF metadata: {}", e), location!())
            })?,
        );

        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        index_writer.add_schema_metadata(
            IVF_METADATA_KEY,
            String::from_utf8(index_ivf_pb.encode_to_vec()).map_err(|e| {
                Error::io(format!("failed to encode IVF metadata: {}", e), location!())
            })?,
        );

        storage_writer.finish().await?;
        index_writer.finish().await?;

        Ok(())
    }
}