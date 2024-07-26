// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow::datatypes::{self, Float32Type, UInt64Type};
use arrow_array::{OffsetSizeTrait, RecordBatch};
use arrow_schema::DataType;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use itertools::Itertools;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result, ROW_ID};
use snafu::{location, Location};

use crate::scalar::{IndexReader, IndexStore};
use crate::vector::graph::OrderedFloat;

use super::index::*;

#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct InvertedIndexBuilder {
    pub(crate) tokens: TokenSet,
    pub(crate) invert_list: InvertedList,
    pub(crate) docs: DocSet,
}

impl InvertedIndexBuilder {
    pub async fn update(
        &mut self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        match new_data.schema().field(0).data_type() {
            DataType::Utf8 => {
                update_index::<i32>(
                    new_data,
                    &mut self.tokens,
                    &mut self.invert_list,
                    &mut self.docs,
                )
                .await?;
            }
            DataType::LargeUtf8 => {
                update_index::<i64>(
                    new_data,
                    &mut self.tokens,
                    &mut self.invert_list,
                    &mut self.docs,
                )
                .await?;
            }
            data_type => {
                return Err(Error::invalid_input(
                    format!("unsupported data type {} for inverted index", data_type),
                    location!(),
                ))
            }
        }

        let token_set_batch = self.tokens.to_batch()?;
        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, token_set_batch.schema())
            .await?;
        token_set_writer.write_record_batch(token_set_batch).await?;
        token_set_writer.finish().await?;

        let invert_list_batch = self.invert_list.to_batch()?;
        let mut invert_list_writer = dest_store
            .new_index_file(INVERT_LIST_FILE, invert_list_batch.schema())
            .await?;
        invert_list_writer
            .write_record_batch(invert_list_batch)
            .await?;
        invert_list_writer.finish().await?;

        let docs_batch = self.docs.to_batch()?;
        let mut docs_writer = dest_store
            .new_index_file(DOCS_FILE, docs_batch.schema())
            .await?;
        docs_writer.write_record_batch(docs_batch).await?;
        docs_writer.finish().await?;

        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let tokens_batch = self.tokens.to_batch()?;
        self.invert_list.remap(mapping);
        let invert_list_batch = self.invert_list.to_batch()?;
        self.docs.remap(mapping);
        let docs_batch = self.docs.to_batch()?;

        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, tokens_batch.schema())
            .await?;
        token_set_writer.write_record_batch(tokens_batch).await?;
        token_set_writer.finish().await?;

        let mut invert_list_writer = dest_store
            .new_index_file(INVERT_LIST_FILE, invert_list_batch.schema())
            .await?;
        invert_list_writer
            .write_record_batch(invert_list_batch)
            .await?;
        invert_list_writer.finish().await?;

        let mut docs_writer = dest_store
            .new_index_file(DOCS_FILE, docs_batch.schema())
            .await?;
        docs_writer.write_record_batch(docs_batch).await?;
        docs_writer.finish().await?;

        Ok(())
    }
}

pub async fn update_index<Offset: OffsetSizeTrait>(
    new_data: SendableRecordBatchStream,
    token_set: &mut TokenSet,
    invert_list: &mut InvertedList,
    docs: &mut DocSet,
) -> Result<()> {
    let mut tokenizer = TOKENIZER.clone();
    let mut stream = new_data;
    while let Some(batch) = stream.try_next().await? {
        let doc_col = batch.column(0).as_string::<Offset>();
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

        for (doc, row_id) in doc_col.iter().zip(row_id_col.iter()) {
            let doc = doc.unwrap();
            let row_id = row_id.unwrap();
            let mut token_stream = tokenizer.token_stream(doc);
            let mut row_token_cnt = HashMap::new();
            let mut token_cnt = 0;
            while let Some(token) = token_stream.next() {
                let token_id = token_set.add(token.text.to_owned());
                row_token_cnt
                    .entry(token_id)
                    .and_modify(|cnt| *cnt += 1)
                    .or_insert(1);
                token_cnt += 1;
            }
            invert_list.add(row_token_cnt, row_id);
            docs.add(row_id, token_cnt);
        }
    }

    Ok(())
}

// InvertedList is a mapping from token ids to row ids
// it's used to retrieve the documents that contain a token
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct InvertedList {
    // the index is the token id
    inverted_list: Vec<PostingList>,
}

impl InvertedList {
    // the schema of the inverted list is | row_id | frequency |
    // and store the offset of
    pub fn to_batch(&self) -> Result<RecordBatch> {
        let batches = self
            .inverted_list
            .iter()
            .map(|list| list.to_batch())
            .collect::<Result<Vec<_>>>()?;

        let offsets = batches
            .iter()
            .scan(0, |offset, batch| {
                *offset += batch.num_rows();
                Some(*offset - batch.num_rows())
            })
            .collect_vec();
        let lengths = self
            .inverted_list
            .iter()
            .map(|list| list.len())
            .collect_vec();
        let metadata = HashMap::from_iter(vec![
            ("offsets".to_owned(), serde_json::to_string(&offsets)?),
            ("lengths".to_owned(), serde_json::to_string(&lengths)?),
        ]);

        let batch =
            concat_batches(batches[0].schema_ref(), batches.iter())?.with_metadata(metadata)?;
        Ok(batch)
    }

    pub async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut inverted_list = Vec::with_capacity(reader.num_rows());
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let row_ids_col = batch[ROW_ID].as_list::<i32>();
            let frequencies_col = batch[FREQUENCY_COL].as_list::<i32>();

            for (row_ids, frequencies) in row_ids_col.iter().zip(frequencies_col.iter()) {
                let row_ids = row_ids.unwrap();
                let frequencies = frequencies.unwrap();
                let row_ids = row_ids.as_primitive::<UInt64Type>().values().to_vec();
                let frequencies = frequencies.as_primitive::<Float32Type>().values().to_vec();
                inverted_list.push(PostingList::new(row_ids, frequencies));
            }
        }

        Ok(Self { inverted_list })
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for list in self.inverted_list.iter_mut() {
            let mut new_row_ids = Vec::new();
            let mut new_freqs = Vec::new();

            for i in 0..list.len() {
                let row_id = list.row_ids[i];
                let freq = list.frequencies[i];

                match mapping.get(&row_id) {
                    Some(Some(new_row_id)) => {
                        new_row_ids.push(*new_row_id);
                        new_freqs.push(freq);
                    }
                    _ => continue,
                }
            }

            *list = PostingList::new(new_row_ids, new_freqs);
        }
    }

    // for efficiency, we don't check if the row_id exists
    // we assume that the row_id is unique and doesn't exist in the list
    pub fn add(&mut self, token_cnt: HashMap<u32, u32>, row_id: u64) {
        for (token_id, freq) in token_cnt {
            let token_id = token_id as usize;
            if token_id >= self.inverted_list.len() {
                self.inverted_list
                    .resize_with(token_id + 1, PostingList::default);
            }
            let list = &mut self.inverted_list[token_id];
            list.row_ids.push(row_id);
            list.frequencies.push(freq as f32);
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, DeepSizeOf)]
pub struct OrderedDoc {
    pub row_id: u64,
    pub score: OrderedFloat,
}

impl OrderedDoc {
    pub fn new(row_id: u64, score: f32) -> Self {
        Self {
            row_id,
            score: OrderedFloat(score),
        }
    }
}

impl PartialOrd for OrderedDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, GenericStringArray, RecordBatch, UInt64Array};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::{FullTextSearchQuery, SargableQuery, ScalarIndex};

    async fn test_inverted_index<Offset: arrow::array::OffsetSizeTrait>() {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, None);

        let mut invert_index = super::InvertedIndexBuilder::default();
        let doc_col = GenericStringArray::<Offset>::from(vec![
            "lance database search",
            "lance database",
            "lance search",
            "database search",
            "unrelated doc",
            "unrelated",
        ]);
        let row_id_col = UInt64Array::from(Vec::from_iter(0..doc_col.len() as u64));
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), false),
                arrow_schema::Field::new(super::ROW_ID, arrow_schema::DataType::UInt64, false),
            ])
            .into(),
            vec![
                Arc::new(doc_col) as ArrayRef,
                Arc::new(row_id_col) as ArrayRef,
            ],
        )
        .unwrap();
        let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
        let stream = Box::pin(stream);

        invert_index
            .update(stream, &store)
            .await
            .expect("failed to update invert index");

        let invert_index = super::InvertedIndex::load(Arc::new(store)).await.unwrap();
        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("lance".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(2));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("database".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(3));
    }

    #[tokio::test]
    async fn test_inverted_index_with_string() {
        test_inverted_index::<i32>().await;
    }

    #[tokio::test]
    async fn test_inverted_index_with_large_string() {
        test_inverted_index::<i64>().await;
    }
}
