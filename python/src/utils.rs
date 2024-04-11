// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{
    cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array, UInt32Array,
};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance::{datatypes::Schema, index::vector::hnsw::builder::*, io::ObjectStore};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::writer::FileWriter;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_linalg::{
    distance::MetricType,
    kmeans::{KMeans as LanceKMeans, KMeansParams},
};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::RT;

#[pyclass(name = "_KMeans")]
pub struct KMeans {
    /// Number of clusters
    k: usize,

    /// Metric type
    metric_type: MetricType,

    max_iters: u32,

    /// A trained KMean model. This is set after calling `fit`.
    trained_kmeans: Option<LanceKMeans<Float32Type>>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (k, metric_type="l2", max_iters=50))]
    fn new(k: usize, metric_type: &str, max_iters: u32) -> PyResult<Self> {
        Ok(Self {
            k,
            metric_type: metric_type.try_into().unwrap(),
            max_iters,
            trained_kmeans: None,
        })
    }

    /// Train the model
    fn fit(&mut self, py: Python, arr: &PyAny) -> PyResult<()> {
        let data = ArrayData::from_pyarrow(arr)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        let params = KMeansParams {
            metric_type: self.metric_type,
            max_iters: self.max_iters,
            ..Default::default()
        };
        let kmeans = RT
            .block_on(
                Some(py),
                LanceKMeans::new_with_params(&fixed_size_arr, self.k, &params),
            )?
            .map_err(|e| PyRuntimeError::new_err(format!("Error training KMeans: {}", e)))?;
        self.trained_kmeans = Some(kmeans);
        Ok(())
    }

    fn predict(&self, py: Python, array: &PyAny) -> PyResult<PyObject> {
        let Some(kmeans) = self.trained_kmeans.as_ref() else {
            return Err(PyRuntimeError::new_err("KMeans must fit (train) first"));
        };
        let data = ArrayData::from_pyarrow(array)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        if kmeans.dimension != fixed_size_arr.value_length() as usize {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: kmean model {} != data {}",
                kmeans.dimension,
                fixed_size_arr.value_length()
            )));
        };
        if !matches!(fixed_size_arr.value_type(), DataType::Float32) {
            return Err(PyValueError::new_err("Must be a FixedSizeList of Float32"));
        };
        let values: Arc<Float32Array> = fixed_size_arr.values().as_primitive().clone().into();
        let membership = RT.block_on(Some(py), kmeans.compute_membership(values))?;
        let cluster_ids: UInt32Array = membership
            .cluster_id_and_distances
            .iter()
            .map(|cd| cd.map(|(c, _)| c))
            .collect();
        cluster_ids.into_data().to_pyarrow(py)
    }

    fn centroids(&self, py: Python) -> PyResult<PyObject> {
        if let Some(kmeans) = self.trained_kmeans.as_ref() {
            let centroids: Float32Array = kmeans.centroids.as_ref().clone();
            let fixed_size_arr =
                FixedSizeListArray::try_new_from_values(centroids, kmeans.dimension as i32)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Error converting centroids to FixedSizeListArray: {}",
                            e
                        ))
                    })?;
            fixed_size_arr.into_data().to_pyarrow(py)
        } else {
            Ok(py.None())
        }
    }
}

#[pyclass(name = "_HNSW")]
pub struct HNSW {
    params: HnswBuildParams,
    hnsw: lance_index::vector::hnsw::HNSW,
    fsl: Arc<dyn Array>,
}

#[pymethods]
impl HNSW {
    #[staticmethod]
    #[pyo3(signature = (
        vectors_array, 
        max_level=7,
        m=20,
        m_max=40,
        ef_construction=100,
        use_select_heuristic=true,
    ))]
    fn build(
        vectors_array: PyAny,
        max_level: u16,
        m: usize,
        m_max: usize,
        ef_construction: usize,
        use_select_heuristic: bool,
    ) -> PyResult<PyObject> {
        let params = HnswBuildParams::default()
            .max_level(max_level)
            .num_edges(m)
            .max_num_edges(m_max)
            .ef_construction(ef_construction)
            .use_select_heuristic(use_select_heuristic);

        let vectors_array = ArrayData::from_pyarrow(&vectors_array)?;
        let vectors_array = FixedSizeListArray::from(data);

        let (hnsw, fsl) = build_hnsw_model(params, vectors_array.downcast::<>())?;
        Ok(Self { params, hnsw, fsl })
    }

    #[pyo3(signature = (file_path))]
    fn to_lance_file(&self, py: Python, file_path: &str) -> PyResult<()> {
        let object_store = ObjectStore::local();
        let path = Path::from_filesystem_path(file_path)?;
        let writer = RT.block_on(Some(py),FileWriter::<ManifestDescribing>::try_new(
            &object_store,
            &path,
            Schema::try_from(self.hnsw.schema().as_ref())?,
            &Default::default(),
        ))??;
        RT.block_on(Some(py), self.hnsw.write(&mut writer))??;
        Ok(())
    }
}
