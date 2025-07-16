// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of the pages_between method in BTreeLookup.
//!
//! This benchmark tests the pages_between method directly, which is responsible for
//! finding which pages in a B-tree index contain values within a given range.
//! It tests various scenarios including:
//! - Small ranges (few pages)
//! - Large ranges (many pages)
//! - Empty ranges (lower_bound >= upper_bound)
//! - Invalid ranges (lower > upper)
//! - Exclusive bounds
//! - Unbounded ranges

use std::{ops::Bound, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen, BatchCount, RowCount};
use lance_index::scalar::btree::{BTreeIndex, OrderableScalarValue};
use lance_index::scalar::ScalarIndex;
use lance_index::Index;
use lance_index::scalar::flat::FlatIndexMetadata;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::Rng;

fn bench_pages_between_direct(c: &mut Criterion) {
    const TOTAL: usize = 1_000_000;
    const BATCH_SIZE: u32 = 4096;

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    // Test different data types
    let test_cases = vec![
        ("int32_random", gen_int32_random(TOTAL)),
        ("float32_random", gen_float32_random(TOTAL)),
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

        // Load the trained index to get access to the BTreeLookup
        let index = rt.block_on(async {
            let index = BTreeIndex::load(store, None).await.unwrap();
            index.prewarm().await.unwrap();
            index
        });

        // Get the BTreeLookup from the index
        let btree_lookup = &index.page_lookup;

        // Benchmark small range query (few pages)
        c.bench_function(format!("pages_between_small_range_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(100000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(101000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark large range query (many pages)
        c.bench_function(format!("pages_between_large_range_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(100000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(800000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark exclusive bounds query
        c.bench_function(format!("pages_between_exclusive_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(100000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(200000)));
                let range = (Bound::Excluded(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark unbounded range query
        c.bench_function(format!("pages_between_unbounded_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(500000)));
                let range = (Bound::Unbounded, Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark empty range query (should return no results)
        c.bench_function(format!("pages_between_empty_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let value = OrderableScalarValue(ScalarValue::Int32(Some(500000)));
                let range = (Bound::Included(&value), Bound::Excluded(&value));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark out of range query (should return no results)
        c.bench_function(format!("pages_between_out_of_range_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(2000000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(3000000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // Benchmark invalid range query (lower > upper)
        c.bench_function(format!("pages_between_invalid_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(600000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(500000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });
    }
}

fn bench_pages_between_edge_cases_direct(c: &mut Criterion) {
    const TOTAL: usize = 100_000; // Smaller dataset for edge case testing
    const BATCH_SIZE: u32 = 4096;

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    // Test different data types
    let test_cases = vec![
        ("int32_random", gen_int32_random(TOTAL)),
        ("float32_random", gen_float32_random(TOTAL)),
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

        // Load the trained index to get access to the BTreeLookup
        let index = rt.block_on(async {
            let index = BTreeIndex::load(store, None).await.unwrap();
            index.prewarm().await.unwrap();
            index
        });

        // Get the BTreeLookup from the index
        let btree_lookup = &index.page_lookup;

        // Benchmark edge cases that specifically test the lower_bound >= upper_bound condition

        // 1. Exact same value for inclusive lower and exclusive upper (empty range)
        c.bench_function(format!("pages_between_empty_range_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let value = OrderableScalarValue(ScalarValue::Int32(Some(50000)));
                let range = (Bound::Included(&value), Bound::Excluded(&value));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // 2. Lower bound greater than upper bound (invalid range)
        c.bench_function(format!("pages_between_invalid_range_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(60000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(50000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // 3. Both bounds exclusive with same value
        c.bench_function(format!("pages_between_exclusive_same_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let value = OrderableScalarValue(ScalarValue::Int32(Some(50000)));
                let range = (Bound::Excluded(&value), Bound::Excluded(&value));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // 4. Range query with values at page boundaries
        c.bench_function(format!("pages_between_page_boundary_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(4096))); // At page boundary
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(8192))); // At next page boundary
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });

        // 5. Range query with negative values (if supported by data type)
        c.bench_function(format!("pages_between_negative_{name}({TOTAL})").as_str(), |b| {
            b.iter(|| {
                let lower = OrderableScalarValue(ScalarValue::Int32(Some(-1000)));
                let upper = OrderableScalarValue(ScalarValue::Int32(Some(1000)));
                let range = (Bound::Included(&lower), Bound::Excluded(&upper));
                black_box(btree_lookup.pages_between(range));
            })
        });
    }
}

// Helper functions to generate different types of test data

fn gen_int32_random(total: usize) -> datafusion::execution::SendableRecordBatchStream {
    let mut rng = rand::thread_rng();
    let values: Vec<i32> = (0..total).map(|_| rng.gen_range(0..total as i32)).collect();
    gen()
        .col(
            "values",
            array::cycle::<arrow::datatypes::Int32Type>(values),
        )
        .col("row_ids", array::step::<arrow::datatypes::UInt64Type>())
        .into_df_stream(RowCount::from(total as u64), BatchCount::from(100))
}

fn gen_float32_random(total: usize) -> datafusion::execution::SendableRecordBatchStream {
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..total)
        .map(|_| rng.gen_range(0.0..total as f32))
        .collect();
    gen()
        .col(
            "values",
            array::cycle::<arrow::datatypes::Float32Type>(values),
        )
        .col("row_ids", array::step::<arrow::datatypes::UInt64Type>())
        .into_df_stream(RowCount::from(total as u64), BatchCount::from(100))
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

criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10));
    targets = bench_pages_between_direct, bench_pages_between_edge_cases_direct);

criterion_main!(benches); 