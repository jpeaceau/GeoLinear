#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "geolinear/geolinear_regressor.h"

namespace py = pybind11;
using namespace geolinear;

PYBIND11_MODULE(_geolinear_cpp, m) {
    m.doc() = "GeoLinear C++ backend — boosted partition-local linear models";

    // ── Config ────────────────────────────────────────────────────────────────
    py::class_<GeoLinearConfig>(m, "GeoLinearConfig")
        .def(py::init<>())
        .def_readwrite("n_rounds",              &GeoLinearConfig::n_rounds)
        .def_readwrite("learning_rate",         &GeoLinearConfig::learning_rate)
        .def_readwrite("y_weight",              &GeoLinearConfig::y_weight)
        .def_readwrite("hvrt_min_samples_leaf", &GeoLinearConfig::hvrt_min_samples_leaf)
        .def_readwrite("hvrt_n_partitions",     &GeoLinearConfig::hvrt_n_partitions)
        .def_readwrite("base_learner",          &GeoLinearConfig::base_learner)
        .def_readwrite("alpha",                 &GeoLinearConfig::alpha)
        .def_readwrite("min_samples_partition", &GeoLinearConfig::min_samples_partition)
        .def_readwrite("hvrt_inner_rounds",        &GeoLinearConfig::hvrt_inner_rounds)
        .def_readwrite("partition_inner_rounds",  &GeoLinearConfig::partition_inner_rounds)
        .def_readwrite("refit_interval",         &GeoLinearConfig::refit_interval)
        .def_readwrite("hvrt_model",             &GeoLinearConfig::hvrt_model)
        .def_readwrite("use_coop_weights",       &GeoLinearConfig::use_coop_weights)
        .def_readwrite("use_t_feature",          &GeoLinearConfig::use_t_feature)
        .def_readwrite("random_state",          &GeoLinearConfig::random_state)
        .def("__repr__", [](const GeoLinearConfig& c) {
            return "<GeoLinearConfig n_rounds=" + std::to_string(c.n_rounds) +
                   " lr=" + std::to_string(c.learning_rate) +
                   " alpha=" + std::to_string(c.alpha) +
                   " y_weight=" + std::to_string(c.y_weight) + ">";
        });

    // ── PartitionCoeffs ───────────────────────────────────────────────────────
    py::class_<GeoLinearBase::PartitionCoeffs>(m, "PartitionCoeffs")
        .def_readonly("partition_id", &GeoLinearBase::PartitionCoeffs::partition_id)
        .def_readonly("coef",         &GeoLinearBase::PartitionCoeffs::coef)
        .def_readonly("intercept",    &GeoLinearBase::PartitionCoeffs::intercept)
        .def_readonly("fallback",     &GeoLinearBase::PartitionCoeffs::fallback)
        .def_readonly("n_samples",    &GeoLinearBase::PartitionCoeffs::n_samples)
        .def("__repr__", [](const GeoLinearBase::PartitionCoeffs& pc) {
            return "<PartitionCoeffs pid=" + std::to_string(pc.partition_id) +
                   " n=" + std::to_string(pc.n_samples) +
                   " fallback=" + (pc.fallback ? "True" : "False") + ">";
        });

    // ── Regressor ─────────────────────────────────────────────────────────────
    py::class_<GeoLinearRegressor>(m, "CppGeoLinearRegressor")
        .def(py::init<GeoLinearConfig>(), py::arg("cfg") = GeoLinearConfig{})
        .def("fit",
             [](GeoLinearRegressor& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoLinearRegressor& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference,
             "Fit the boosted partition-local Ridge model.")
        .def("predict",
             [](const GeoLinearRegressor& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict(X);
             },
             py::arg("X"),
             "Predict continuous values. Returns ndarray (n,).")
        .def("is_fitted",    &GeoLinearRegressor::is_fitted)
        .def("intercept",    &GeoLinearRegressor::intercept)
        .def("n_stages",     &GeoLinearRegressor::n_stages)
        .def("get_stage_coeffs",
             &GeoLinearRegressor::get_stage_coeffs,
             py::arg("stage_idx"))
        .def("get_stage_partition_ids",
             &GeoLinearRegressor::get_stage_partition_ids,
             py::arg("stage_idx"))
        .def("apply_stage",
             [](const GeoLinearRegressor& self, int stage_idx,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.apply_stage(stage_idx, X);
             },
             py::arg("stage_idx"), py::arg("X"))
        .def("__repr__", [](const GeoLinearRegressor& r) {
            return std::string("<CppGeoLinearRegressor fitted=") +
                   (r.is_fitted() ? "True" : "False") +
                   " stages=" + std::to_string(r.n_stages()) + ">";
        });

    // ── Classifier ────────────────────────────────────────────────────────────
    py::class_<GeoLinearClassifier>(m, "CppGeoLinearClassifier")
        .def(py::init<GeoLinearConfig>(), py::arg("cfg") = GeoLinearConfig{})
        .def("fit",
             [](GeoLinearClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoLinearClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference,
             "Fit the boosted logistic classifier (y must be 0/1 float).")
        .def("predict",
             [](const GeoLinearClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict(X);
             },
             py::arg("X"),
             "Predict 0/1 labels. Returns ndarray (n,).")
        .def("predict_proba",
             [](const GeoLinearClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict_proba(X);
             },
             py::arg("X"),
             "Predict class-1 probabilities (sigmoid of raw scores). Returns ndarray (n,).")
        .def("is_fitted",    &GeoLinearClassifier::is_fitted)
        .def("n_stages",     &GeoLinearClassifier::n_stages)
        .def("get_stage_coeffs",
             &GeoLinearClassifier::get_stage_coeffs,
             py::arg("stage_idx"))
        .def("get_stage_partition_ids",
             &GeoLinearClassifier::get_stage_partition_ids,
             py::arg("stage_idx"))
        .def("apply_stage",
             [](const GeoLinearClassifier& self, int stage_idx,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.apply_stage(stage_idx, X);
             },
             py::arg("stage_idx"), py::arg("X"))
        .def("__repr__", [](const GeoLinearClassifier& c) {
            return std::string("<CppGeoLinearClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") +
                   " stages=" + std::to_string(c.n_stages()) + ">";
        });
}
