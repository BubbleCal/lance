// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use bytes::Bytes;
use futures::{stream::TryStreamExt, StreamExt};
use itertools::Itertools;
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_file::v2::{
    reader::FileReader,
    writer::{FileWriter, FileWriterOptions},
};
use lance_index::{
    pb::Ivf as PbIvf,
    vector::{
        graph::{DISTS_FIELD, NEIGHBORS_FIELD},
        hnsw::{builder::HNSW_METADATA_KEY, HnswMetadata, VECTOR_ID_FIELD},
        ivf::{
            shuffler::IvfShuffler,
            storage::{IvfData, IVF_PARTITION_KEY},
            IvfBuildParams,
        },
        quantizer::Quantizer,
        v3::{shuffler::IvfShuffleReader, subindex::IvfSubIndex},
    },
    IndexMetadata, INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY,
};
use lance_io::{object_store::ObjectStore, stream::RecordBatchStreamAdapter, ReadBatchParams};
use lance_linalg::distance::DistanceType;
use lance_table::{format::SelfDescribingFileReader, io::manifest::ManifestDescribing};
use object_store::path::Path;
use prost::Message;
use serde_json::json;
use snafu::{location, Location};
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;

use crate::Dataset;

use super::{build_ivf_model, io::build_hnsw_quantization_partition, scan_index_field_stream, Ivf};

pub struct IvfIndexSpec<I: IvfSubIndex> {
    dataset: Dataset,
    column: String,
    index_dir: String,
    distance_type: DistanceType,
    ivf_params: IvfBuildParams,
    shuffler: IvfShuffler, // TODO: replace with trait IvfShuffler
    sub_index: I,
    quantizer: Quantizer,

    temp_dir: TempDir,
    temp_path: Option<String>,
}

impl<I: IvfSubIndex> IvfIndexSpec<I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Dataset,
        column: String,
        index_dir: String,
        distance_type: DistanceType,
        ivf_params: IvfBuildParams,
        shuffler: IvfShuffler,
        sub_index: I,
        quantizer: Quantizer,
        temp_path: Option<String>,
    ) -> Self {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        Self {
            dataset,
            column,
            index_dir,
            distance_type,
            ivf_params,
            shuffler,
            sub_index,
            quantizer,
            temp_dir,
            temp_path,
        }
    }

    pub async fn build(self) -> Result<()> {
        // step 1: train the ivf model and the scalar quantizer
        let (ivf_model, sq) = self.build_or_load_models().await?;

        // prepare the ivf transformer
        let ivf_transformer = lance_index::vector::ivf::new_ivf_with_quantizer(
            ivf_model.centroids.values(),
            self.distance_type,
            &self.column,
            self.quantizer.clone(),
            None,
        )?;

        // step 2: build the ivf partitions to disk
        let ivf_reader = self
            .build_ivf_partitions_to_disk(ivf_transformer.clone())
            .await?;

        // step 3: build the hnsw partitions -- build larger partitions first
        // [(partition_size, partition_id), ...]
        let partition_build_order = (0..ivf_model.num_partitions())
            .map(|partition_id| ivf_reader.partiton_size(partition_id))
            .collect::<Result<Vec<_>>>()?
            // sort by partition size in descending order
            .into_iter()
            .enumerate()
            .map(|(idx, x)| (x, idx))
            .sorted()
            .rev()
            .map(|(_, idx)| idx)
            .collect::<Vec<_>>();

        // open writers for the files -- for schema
        let (index_writer, aux_writer) = self
            .make_index_and_aux_writer(self.quantizer.clone())
            .await?;
        let index_schema = index_writer.schema().clone();
        let aux_schema = aux_writer.schema().clone();

        let mut res = vec![];
        for &partition in &partition_build_order {
            let ivf_reader = ivf_reader.clone();
            let index_schema = index_schema.clone();
            let aux_schema = aux_schema.clone();
            let quantizer = self.quantizer.clone();
            let length = self
                .build_hnsw(ivf_reader, partition, index_schema, aux_schema, quantizer)
                .await?;
            res.push(length);
        }

        // step 4: write the final index
        let written_hnsw_sizes = partition_build_order
            .into_iter()
            .zip(res.into_iter())
            .sorted()
            .map(|(_, length)| length)
            .collect::<Vec<_>>();
        self.write_aggregate_file(
            ivf_model,
            self.quantizer.clone(),
            written_hnsw_sizes,
            index_writer,
            aux_writer,
        )
        .await
    }

    async fn make_tmp_object_store(&self) -> Result<(ObjectStore, Path)> {
        let path = self.temp_path.clone().unwrap_or_else(|| {
            self.temp_dir
                .path()
                .to_str()
                .expect("non-UTF8 path")
                .to_string()
        });
        ObjectStore::from_path(&path)
    }

    async fn build_ivf_partitions_to_disk(
        &self,
        ivf_transformer: lance_index::vector::ivf::Ivf,
    ) -> Result<Arc<dyn IvfShuffleReader>> {
        let (object_store, base_path) = self.make_tmp_object_store().await?;
        let output_dir = base_path.child("shuffle");

        // check success tag
        let success_tag = self
            .get_file_from_temp(&["shuffle", "__SUCCESS"])
            .await
            .ok()
            .filter(|x| x.to_vec() == b"shuffle_complete");
        // can't chain functionally because we need to await
        if success_tag.is_some() {
            if let Ok(reader) = MaterailizingShuffleReader::new_from_dir(
                object_store.clone(),
                output_dir.clone(),
                self.ivf_params.num_partitions,
            )
            .await
            {
                log::info!("Shuffle already complete, loading shuffle from disk");
                return Ok(Arc::new(reader));
            }
        }

        let stream = scan_index_field_stream(&self.dataset, &self.column).await?;

        let mut transformed_stream = Box::pin(
            stream
                .map(move |batch| {
                    let ivf_transformer = ivf_transformer.clone();
                    tokio::spawn(async move { ivf_transformer.transform(&batch?).await })
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

        let shuffler = MaterailizingShuffler::new(
            object_store,
            output_dir,
            self.ivf_params.num_partitions,
            4096,
        );

        let shuffle_reader = shuffler
            .shuffle(Box::new(RecordBatchStreamAdapter::new(
                schema,
                transformed_stream,
            )))
            .await?;

        self.put_file_to_temp(&["shuffle", "__SUCCESS"], b"shuffle_complete")
            .await?;
        Ok(shuffle_reader)
    }

    fn dims(&self) -> Result<usize> {
        let schema = self.dataset.schema().clone();
        let field = schema.field(&self.column).ok_or_else(|| Error::Schema {
            message: format!("column {} does not exist in data stream", &self.column),
            location: location!(),
        })?;

        match field.data_type() {
            DataType::FixedSizeList(_, dim) => Ok(dim as usize),
            _ => Err(Error::Schema {
                message: format!("column {} is not a fixed size list", &self.column),
                location: location!(),
            }),
        }
    }

    async fn make_index_and_aux_writer(
        &self,
        quantizer: Quantizer, // need this to write metadata about quantization config
    ) -> Result<(FileWriter, FileWriter)> {
        let object_store = self.dataset.object_store();
        let path = self
            .dataset
            .indices_dir()
            .child(self.uuid.clone())
            .child(INDEX_FILE_NAME);
        let writer = object_store.create(&path).await?;

        let schema = Schema::new(vec![
            VECTOR_ID_FIELD.clone(),
            NEIGHBORS_FIELD.clone(),
            DISTS_FIELD.clone(),
        ]);
        let schema = lance_core::datatypes::Schema::try_from(&schema)?;
        let mut writer = FileWriter::try_new(
            writer,
            path.to_string(),
            schema,
            FileWriterOptions::default(),
        )?;
        writer.add_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            json!(IndexMetadata {
                index_type: format!("IVF_HNSW_{}", quantizer.quantization_type()),
                distance_type: self.distance_type.to_string(),
            })
            .to_string()
            .as_str(),
        );

        let aux_path = self
            .dataset
            .indices_dir()
            .child(self.uuid.clone())
            .child(INDEX_AUXILIARY_FILE_NAME);
        let aux_writer = object_store.create(&aux_path).await?;
        let schema = Schema::new(vec![
            ROW_ID_FIELD.clone(),
            arrow_schema::Field::new(
                quantizer.column(),
                DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true)),
                    quantizer.code_dim() as i32,
                ),
                false,
            ),
        ]);
        let schema = lance_core::datatypes::Schema::try_from(&schema)?;
        let mut aux_writer = FileWriter::try_new(
            aux_writer,
            aux_path.to_string(),
            schema,
            FileWriterOptions::default(),
        )?;
        aux_writer.add_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            json!(IndexMetadata {
                index_type: quantizer.quantization_type().to_string(),
                distance_type: self.distance_type.to_string(),
            })
            .to_string()
            .as_str(),
        );

        // TODO: FOR PQ write the code book in aux metadata
        Ok((writer, aux_writer))
    }

    async fn make_parted_index_and_aux_writer(
        &self,
        partition_id: usize,
        // These should be inferrable from index params
        // TODO: remove these
        part_schema: lance_core::datatypes::Schema,
        aux_schema: lance_core::datatypes::Schema,
    ) -> Result<(FileWriter, FileWriter)> {
        let (object_store, base_path) = self.make_tmp_object_store().await?;
        let output_dir = base_path.child("index_parts");

        let (part_file, aux_part_file) = (
            output_dir.child(format!("part_{}", partition_id)),
            output_dir.child(format!("part_aux_{}", partition_id)),
        );

        let part_writer = FileWriter::try_new(
            &object_store,
            part_file.to_string(),
            part_schema,
            Default::default(),
        )
        .await?;

        let aux_part_writer = FileWriter::try_new(
            &object_store,
            aux_part_file.to_string(),
            aux_schema,
            Default::default(),
        )
        .await?;

        Ok((part_writer, aux_part_writer))
    }

    async fn make_parted_index_and_aux_reader(
        &self,
        partition_id: usize,
    ) -> Result<(FileReader, FileReader)> {
        let (object_store, base_path) = self.make_tmp_object_store().await?;
        let output_dir = base_path.child("index_parts");
        let (part_file, aux_part_file) = (
            output_dir.child(format!("part_{}", partition_id)),
            output_dir.child(format!("part_aux_{}", partition_id)),
        );

        let part_reader =
            FileReader::try_new_self_described(&object_store, &part_file, None).await?;

        let aux_reader =
            FileReader::try_new_self_described(&object_store, &aux_part_file, None).await?;

        Ok((part_reader, aux_reader))
    }

    async fn build_hnsw(
        &self,
        ivf_reader: Arc<dyn IvfShuffleReader>,
        partition_id: usize,
        // These should be inferrable from index params
        // TODO: remove these
        part_schema: lance_core::datatypes::Schema,
        aux_schema: lance_core::datatypes::Schema,
        quantizer: Quantizer,
    ) -> Result<usize> {
        if ivf_reader.partiton_size(partition_id)? == 0 {
            return Ok(0);
        }

        log::info!("Loading vectors for HNSW partition {}", partition_id);

        let dataset = Arc::new(self.dataset.clone());
        let column = Arc::new(self.column.to_owned());
        let mut params = self.hnsw_params.clone();
        params.parallel_limit = Some(num_cpus::get() - 2);
        let hnsw_params = Arc::new(params);

        let (index_writer, aux_writer) = self
            .make_parted_index_and_aux_writer(partition_id, part_schema, aux_schema)
            .await?;

        // load the ivf partition
        let stream = ivf_reader
            .read_partition(partition_id)
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let row_id_array = stream
            .iter()
            .map(|batch| batch.column_by_name(ROW_ID).unwrap())
            .cloned()
            .collect::<Vec<_>>();
        let vector_arrs = stream
            .iter()
            .map(|batch| batch.column_by_name(&self.column).unwrap())
            .cloned()
            .collect::<Vec<_>>();
        let vectors = arrow_select::concat::concat(
            vector_arrs
                .iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        std::mem::drop(vector_arrs);
        std::mem::drop(stream);

        log::info!(
            "Vectors loading starting build HNSW partition {}",
            partition_id
        );

        build_hnsw_quantization_partition(
            dataset,
            column,
            self.distance_type,
            hnsw_params,
            index_writer,
            Some(aux_writer),
            quantizer,
            row_id_array,
            vec![], // no sq codes needed as they get computed right before writing
            Some(vectors),
        )
        .await
    }

    async fn write_aggregate_file(
        &self,
        mut ivf_model: Ivf,
        quantizer: Quantizer,
        partition_lengths: Vec<usize>,
        mut index_writer: FileWriter<ManifestDescribing>,
        mut aux_writer: FileWriter<ManifestDescribing>,
    ) -> Result<()> {
        let mut aux_ivf = IvfData::empty();
        let mut hnsw_metadata = Vec::with_capacity(ivf_model.num_partitions());
        for (part_id, length) in partition_lengths.into_iter().enumerate() {
            let offset = index_writer.tell().await?;

            if length == 0 {
                ivf_model.add_partition(offset, 0);
                aux_ivf.add_partition(0);
                hnsw_metadata.push(HnswMetadata::default());
                continue;
            }

            let (part_reader, aux_part_reader) =
                self.make_parted_index_and_aux_reader(part_id).await?;
            let batches = futures::stream::iter(0..part_reader.num_batches())
                .map(|batch_id| {
                    part_reader.read_batch(
                        batch_id as i32,
                        ReadBatchParams::RangeFull,
                        part_reader.schema(),
                        None,
                    )
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;
            index_writer.write(&batches).await?;
            ivf_model.add_partition(offset, length as u32);
            hnsw_metadata.push(serde_json::from_str(
                part_reader.schema().metadata[HNSW_METADATA_KEY].as_str(),
            )?);
            std::mem::drop(part_reader);

            let batches = futures::stream::iter(0..aux_part_reader.num_batches())
                .map(|batch_id| {
                    aux_part_reader.read_batch(
                        batch_id as i32,
                        ReadBatchParams::RangeFull,
                        aux_part_reader.schema(),
                        None,
                    )
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;
            std::mem::drop(aux_part_reader);

            let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            aux_writer.write(&batches).await?;
            aux_ivf.add_partition(num_rows as u32);
        }

        let hnsw_metadata_json = json!(hnsw_metadata);
        index_writer.add_metadata(IVF_PARTITION_KEY, &hnsw_metadata_json.to_string());
        let mut ivf_data = IvfData::with_centroids(ivf_model.centroids.clone());

        for length in ivf_model.lengths {
            ivf_data.add_partition(length);
        }
        ivf_data.write(&mut index_writer).await?;

        index_writer.finish().await?;

        // Write the aux file
        aux_ivf.write(&mut aux_writer).await?;
        aux_writer.add_metadata(
            quantizer.metadata_key(),
            quantizer.metadata(None)?.to_string().as_str(),
        );
        aux_writer.finish().await?;
        Ok(())
    }

    async fn put_file_to_temp(&self, file_path: &[&str], data: &[u8]) -> Result<()> {
        let (object_store, base_path) = self.make_tmp_object_store().await?;

        let mut output_path = base_path;
        for &part in file_path {
            output_path = output_path.child(part);
        }

        let mut file = object_store.create(&output_path).await?;
        file.write_all(data).await?;
        file.flush().await?;
        file.shutdown().await?;

        Ok(())
    }

    async fn get_file_from_temp(&self, file_path: &[&str]) -> Result<Bytes> {
        let (object_store, base_path) = self.make_tmp_object_store().await?;

        let mut output_path = base_path;
        for &part in file_path {
            output_path = output_path.child(part);
        }

        let file = object_store.open(&output_path).await?;

        file.get_range(0..file.size().await?).await
    }

    async fn build_or_load_models(&self) -> Result<Ivf> {
        // try load the models from disk
        let loaded_ivf_model: Option<Ivf> = self
            .get_file_from_temp(&["stages", "ivf.pb"])
            .await
            .ok()
            .and_then(|x| PbIvf::decode(x).ok())
            .and_then(|x| (&x).try_into().ok());

        if let Some(ivf_model) = loaded_ivf_model {
            log::info!("Loaded IVF and SQ models from disk");
            // copy only the centroids over -- drop the offsets
            let ivf_model = Ivf::new(ivf_model.centroids);
            return Ok(ivf_model);
        }

        // train the ivf model and the scalar quantizer
        let ivf_model = build_ivf_model(
            &self.dataset,
            &self.column,
            self.quantizer.dim(),
            self.distance_type,
            &self.ivf_params,
        )
        .await?;

        let mut cloned_ivf_model = ivf_model.clone();
        // populate the ivf model with dummy partitions
        (0..cloned_ivf_model.num_partitions()).for_each(|_| cloned_ivf_model.add_partition(0, 0));
        // write the models to disk
        let pb_ivf: PbIvf = (&cloned_ivf_model).try_into()?;

        self.put_file_to_temp(&["stages", "ivf.pb"], &pb_ivf.encode_to_vec())
            .await?;

        Ok(ivf_model)
    }
}
