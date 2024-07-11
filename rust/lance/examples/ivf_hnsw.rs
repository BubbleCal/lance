// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use std::{collections::HashSet, sync::Arc};

use arrow::{
    array::AsArray,
    datatypes::{Int32Type, UInt64Type},
};
use arrow_array::{types::Float32Type, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures::TryStreamExt;
use itertools::Itertools;
use lance::index::{scalar::ScalarIndexParams, vector::VectorIndexParams};
use lance::Dataset;
use lance_core::ROW_ID;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::MetricType;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset URI
    uri: String,

    /// Vector column name
    #[arg(short, long, value_name = "NAME", default_value = "vector")]
    column: Option<String>,

    #[arg(long, default_value = None)]
    filter: Option<String>,

    #[arg(long, default_value = "150")]
    ef: usize,

    /// Max number of edges of each node.
    #[arg(long, default_value = "15")]
    num_edges: usize,

    #[arg(long, default_value = "7")]
    max_level: u16,

    #[arg(long, default_value = "32")]
    nlist: usize,

    #[arg(long, default_value = "1")]
    nprobe: usize,

    #[arg(short, default_value = "10")]
    k: usize,

    #[arg(long, default_value = "false")]
    create_index: bool,

    #[arg(long, default_value = "cosine")]
    metric_type: String,
}

#[cfg(test)]
fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..mat.num_rows() {
        let dist = lance_linalg::distance::dot_distance(query, mat.row(i).unwrap());
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    let mut dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());
    // let bitmap_params = ScalarIndexParams {
    //     force_index_type: Some(lance::index::scalar::ScalarIndexType::Bitmap),
    // };
    // println!("creating bitmap index");
    // dataset
    //     .create_index(&["genres"], IndexType::Scalar, None, &bitmap_params, true)
    //     .await
    //     .unwrap();
    // println!("create bitmap index done");
    // return;

    let column = args.column.as_deref().unwrap_or("vector");
    let metric_type = MetricType::try_from(args.metric_type.as_str()).unwrap();

    let mut ivf_params = IvfBuildParams::new(args.nlist);
    ivf_params.sample_rate = 40480;
    let hnsw_params = HnswBuildParams::default()
        .ef_construction(args.ef)
        .num_edges(args.num_edges);
    let mut pq_params = SQBuildParams::default();
    pq_params.sample_rate = 40480;
    let params =
        VectorIndexParams::with_ivf_hnsw_sq_params(metric_type, ivf_params, hnsw_params, pq_params);
    println!("{:?}", params);

    if args.create_index {
        let now = std::time::Instant::now();
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        println!("build={:.3}s", now.elapsed().as_secs_f32());
    }

    let num_rows = dataset.count_rows(None).await.unwrap() as u64;
    println!("Loaded {} records", num_rows);

    let indice = 1023;
    println!("query with {}-th row vector", indice);
    let q = dataset
        .take(&[indice], &dataset.schema().project(&[column]).unwrap())
        .await
        .unwrap()
        .column(0)
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .clone();

    let columns: &[&str] = &[];
    let gt_batch = dataset
        .scan()
        .project(columns)
        .unwrap()
        .filterv2(&args.filter)
        .unwrap()
        .prefilter(true)
        .with_row_id()
        .flat_nearest(column, &q, args.k)
        .unwrap()
        .distance_metric(metric_type)
        .try_into_batch()
        .await
        .unwrap();
    // let years = gt_batch
    //     .column_by_name("year")
    //     .unwrap()
    //     .as_primitive::<Int32Type>();
    // for v in years.values() {
    //     assert!(*v > 2015);
    // }
    let gt_results = gt_batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .into_iter()
        .map(|x| x.unwrap())
        .collect::<HashSet<_>>();
    println!("gt num: {}", gt_results.len());

    let mut scan = dataset.scan();
    let plan = scan
        .project(columns)
        .unwrap()
        .filterv2(&args.filter)
        .unwrap()
        .prefilter(true)
        // .fast_search()
        .with_row_id()
        .nearest(column, &q, args.k)
        .unwrap()
        .nprobs(args.nprobe);
    println!("{:?}", plan.explain_plan(true).await.unwrap());

    let now = std::time::Instant::now();
    let results = plan.try_into_batch().await.unwrap();
    // let years = results
    //     .column_by_name("year")
    //     .unwrap()
    //     .as_primitive::<Int32Type>();
    // for v in years.values() {
    //     assert!(*v > 2015);
    // }
    let results = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .into_iter()
        .map(|x| x.unwrap())
        .collect::<HashSet<_>>();
    println!("results num: {}", results.len());
    let recall =
        gt_results.intersection(&results).collect_vec().len() as f32 / gt_results.len() as f32;
    println!(
        "level={}, nprobe={}, k={}, search={:?}, recall={}",
        args.max_level,
        args.nprobe,
        args.k,
        now.elapsed(),
        recall,
    );

    for concurrency in [1, 2, 4, 8, 16, 32, 64] {
        // execute query with `concurrency` threads for 5 seconds and calculate qps and collect latencies
        let threads = (0..concurrency)
            .map(|_| {
                let column = column.to_owned();
                let q = q.clone();
                let dataset = dataset.clone();
                let filter = args.filter.clone();
                tokio::task::spawn(async move {
                    let mut scan = dataset.scan();
                    let plan = scan
                        .project(columns)
                        .unwrap()
                        .filterv2(&filter)
                        .unwrap()
                        .prefilter(true)
                        .fast_search()
                        .with_row_id()
                        .nearest(&column, &q, args.k)
                        .unwrap()
                        .nprobs(args.nprobe);
                    let mut qps = 0;
                    let mut latencies = Vec::with_capacity(10000);
                    let now = std::time::Instant::now();
                    while now.elapsed().as_secs() < 10 {
                        let now = std::time::Instant::now();
                        plan.try_into_stream()
                            .await
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap();
                        latencies.push(now.elapsed());
                        qps += 1;
                    }
                    (qps, latencies)
                })
            })
            .collect::<Vec<_>>();
        let mut qps = 0;
        let mut latencies = Vec::with_capacity(10000);
        for t in threads {
            let (q, l) = t.await.unwrap();
            qps += q;
            latencies.extend(l);
        }
        qps /= 10;
        latencies.sort();
        let p90 = latencies[latencies.len() * 90 / 100 - 1];
        let p95 = latencies[latencies.len() * 95 / 100 - 1];
        let p99 = latencies[latencies.len() * 99 / 100 - 1];
        let mean = latencies.iter().sum::<std::time::Duration>() / latencies.len() as u32;
        println!(
            "level={}, nprobe={}, k={}, concurrency={}, qps={}, mean={:?}, p90={:?}, p95={:?}, p99={:?}",
            args.max_level, args.nprobe, args.k, concurrency, qps, mean, p90, p95, p99,
        );
    }
    println!(
        "warm up: level={}, nprobe={}, k={}, search={:?}",
        args.max_level,
        args.nprobe,
        args.k,
        now.elapsed().div_f32(10.0),
    );
}
