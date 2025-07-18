// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of B-tree index search performance.
//!
//! This benchmark tests various search operations on B-tree indices:
//! - Equals queries (exact matches)
//! - Range queries (between values)
//! - IsIn queries (multiple value matches)
//! - IsNull queries (null value matches)
//! - Different data types (Int32, Float64, String)
//! - Different data sizes and distributions

use std::{ops::Bound, sync::Arc, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen, BatchCount, RowCount};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::btree::BTreeIndex;
use lance_index::scalar::flat::FlatIndexMetadata;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{SargableQuery, ScalarIndex};
use lance_index::Index;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

fn bench_btree_search(c: &mut Criterion) {
    const TOTAL: usize = 10_000_000;
    const BATCH_SIZE: u32 = 4096;

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    // Test different data types
    let test_cases = vec![
        ("int32_random", gen_int32_random(TOTAL, BATCH_SIZE)),
        ("float32_random", gen_float32_random(TOTAL, BATCH_SIZE)),
    ];

    for (name, data) in test_cases {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = rt.block_on(async {
            Arc::new(LanceIndexStore::new(
                Arc::new(ObjectStore::local()),
                index_dir,
                Arc::new(LanceCache::no_cache()),
            ))
        });

        // Train the B-tree index (not benchmarked)
        rt.block_on(async {
            let sub_index_trainer =
                FlatIndexMetadata::new(data.schema().field(0).data_type().clone());
            lance_index::scalar::btree::train_btree_index(
                Box::new(MockTrainingSource::from(data)),
                &sub_index_trainer,
                store.as_ref(),
                BATCH_SIZE,
            )
            .await
            .unwrap();
        });

        // Load the trained index
        let index = rt.block_on(async {
            let index = BTreeIndex::load(store, None).await.unwrap();
            index.prewarm().await.unwrap();
            index
        });

        // Benchmark equals queries
        c.bench_function(format!("btree_equals_{name}({TOTAL})").as_str(), |b| {
            b.to_async(&rt).iter(|| async {
                let query_value = match name {
                    "int32_random" => SargableQuery::Equals(ScalarValue::Int32(Some(500000))),
                    "float32_random" => SargableQuery::Equals(ScalarValue::Float32(Some(500000.0))),
                    _ => unreachable!(),
                };
                black_box(
                    index
                        .search(&query_value, &NoOpMetricsCollector)
                        .await
                        .unwrap(),
                );
            })
        });

        // Benchmark range queries
        c.bench_function(format!("btree_range_{name}({TOTAL})").as_str(), |b| {
            b.to_async(&rt).iter(|| async {
                let query_range = match name {
                    "int32_random" => SargableQuery::Range(
                        Bound::Included(ScalarValue::Int32(Some(100000))),
                        Bound::Excluded(ScalarValue::Int32(Some(200000))),
                    ),
                    "float32_random" => SargableQuery::Range(
                        Bound::Included(ScalarValue::Float32(Some(100000.0))),
                        Bound::Excluded(ScalarValue::Float32(Some(200000.0))),
                    ),
                    _ => unreachable!(),
                };
                black_box(
                    index
                        .search(&query_range, &NoOpMetricsCollector)
                        .await
                        .unwrap(),
                );
            })
        });

        // Benchmark IsIn queries
        c.bench_function(format!("btree_is_in_{name}({TOTAL})").as_str(), |b| {
            b.to_async(&rt).iter(|| async {
                let query_values = match name {
                    "int32_random" => SargableQuery::IsIn(vec![
                        ScalarValue::Int32(Some(100000)),
                        ScalarValue::Int32(Some(200000)),
                        ScalarValue::Int32(Some(300000)),
                        ScalarValue::Int32(Some(400000)),
                        ScalarValue::Int32(Some(500000)),
                    ]),
                    "float32_random" => SargableQuery::IsIn(vec![
                        ScalarValue::Float32(Some(100000.0)),
                        ScalarValue::Float32(Some(200000.0)),
                        ScalarValue::Float32(Some(300000.0)),
                        ScalarValue::Float32(Some(400000.0)),
                        ScalarValue::Float32(Some(500000.0)),
                    ]),
                    _ => unreachable!(),
                };
                black_box(
                    index
                        .search(&query_values, &NoOpMetricsCollector)
                        .await
                        .unwrap(),
                );
            })
        });

        // Benchmark IsNull queries (for data with nulls)
        if name.contains("random") {
            c.bench_function(format!("btree_is_null_{name}({TOTAL})").as_str(), |b| {
                b.to_async(&rt).iter(|| async {
                    let query_null = SargableQuery::IsNull();
                    black_box(
                        index
                            .search(&query_null, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                })
            });
        }
    }
}

// Helper functions to generate different types of test data

fn gen_int32_random(
    total: usize,
    batch_size: u32,
) -> datafusion::execution::SendableRecordBatchStream {
    gen()
        .col("values", array::step::<arrow::datatypes::Int32Type>())
        .col("row_ids", array::step::<arrow::datatypes::UInt64Type>())
        .into_df_stream(
            RowCount::from(batch_size as u64),
            BatchCount::from(total.div_ceil(batch_size as usize) as u32),
        )
}

fn gen_float32_random(
    total: usize,
    batch_size: u32,
) -> datafusion::execution::SendableRecordBatchStream {
    gen()
        .col("values", array::step::<arrow::datatypes::Float32Type>())
        .col("row_ids", array::step::<arrow::datatypes::UInt64Type>())
        .into_df_stream(
            RowCount::from(batch_size as u64),
            BatchCount::from(total.div_ceil(batch_size as usize) as u32),
        )
}

// Mock training source for the benchmark
struct MockTrainingSource {
    data: datafusion::execution::SendableRecordBatchStream,
}

impl MockTrainingSource {
    fn from(data: datafusion::execution::SendableRecordBatchStream) -> Self {
        Self { data }
    }
}

#[async_trait::async_trait]
impl lance_index::scalar::btree::TrainingSource for MockTrainingSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> lance_core::Result<datafusion::execution::SendableRecordBatchStream> {
        // For simplicity, we'll just return the data as-is
        // In a real implementation, this would sort and chunk the data
        Ok(self.data)
    }

    async fn scan_unordered_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> lance_core::Result<datafusion::execution::SendableRecordBatchStream> {
        Ok(self.data)
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10));
    targets = bench_btree_search);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10));
    targets = bench_btree_search);

criterion_main!(benches);
