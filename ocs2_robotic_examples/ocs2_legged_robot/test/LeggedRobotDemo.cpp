#include <iostream>
#include <glog/logging.h>

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_core/misc/LinearAlgebra.h>

#include "ocs2_legged_robot/common/ModelSettings.h"
#include "ocs2_legged_robot/common/Types.h"
#include "ocs2_legged_robot/constraint/ZeroForceConstraint.h"
#include "ocs2_legged_robot/constraint/EndEffectorLinearConstraint.h"
#include "ocs2_legged_robot/constraint/FrictionConeConstraint.h"
#include "ocs2_legged_robot/test/AnymalFactoryFunctions.h"


//////////////////////////////////////////// TestZeroForceConstraint ////////////////////////////////////////////////////

int TestZeroForceConstraintEvaluate() {
  const ocs2::legged_robot::ModelSettings modelSettings;  // default constructor just to get contactNames3DoF
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  const std::shared_ptr<ocs2::legged_robot::SwitchedModelReferenceManager> referenceManagerPtr = ocs2::legged_robot::createReferenceManager(centroidalModelInfo.numThreeDofContacts);
  ocs2::PreComputation preComputation;
  for (size_t i = 0; i < centroidalModelInfo.numThreeDofContacts; i++) {
    ocs2::legged_robot::ZeroForceConstraint zeroForceConstraint(*referenceManagerPtr, i, centroidalModelInfo);

    // evaluation point
    const ocs2::scalar_t t = 0.0;
    const ocs2::vector_t u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);
    const ocs2::vector_t x = ocs2::vector_t::Random(centroidalModelInfo.stateDim);
    const ocs2::legged_robot::vector3_t eeForce = ocs2::centroidal_model::getContactForces(u, i, centroidalModelInfo);

    const auto value = zeroForceConstraint.getValue(t, x, u, preComputation);
    const auto approx = zeroForceConstraint.getLinearApproximation(t, x, u, preComputation);

    LOG(INFO) << "Contact: " << modelSettings.contactNames3DoF[i];
    LOG(INFO) << "Value\n" << value.transpose();
    LOG(INFO) << "LinearApproximation\n" << approx;
    LOG(INFO) << value.isApprox(approx.f);
    LOG(INFO) << approx.dfdx.isApproxToConstant(0.0);
    LOG(INFO) << eeForce.isApprox(approx.dfdu * u);
  }
  return 0;
}

int TestZeroForceConstraintClone() {
  const ocs2::legged_robot::ModelSettings modelSettings;  // default constructor just to get contactNames3DoF
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  const std::shared_ptr<ocs2::legged_robot::SwitchedModelReferenceManager> referenceManagerPtr = ocs2::legged_robot::createReferenceManager(centroidalModelInfo.numThreeDofContacts);
  ocs2::PreComputation preComputation;
  constexpr size_t eeIndex = 0;
  ocs2::legged_robot::ZeroForceConstraint zeroForceConstraint(*referenceManagerPtr, eeIndex, centroidalModelInfo);
  std::unique_ptr<ocs2::legged_robot::ZeroForceConstraint> zeroForceConstraintPtr(zeroForceConstraint.clone());

  // evaluation point
  const ocs2::scalar_t t = 0.0;
  const ocs2::vector_t u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);
  const ocs2::vector_t x = ocs2::vector_t::Random(centroidalModelInfo.stateDim);

  const auto value = zeroForceConstraint.getValue(t, x, u, preComputation);
  const auto cloneValue = zeroForceConstraintPtr->getValue(t, x, u, preComputation);

  const auto approx = zeroForceConstraint.getLinearApproximation(t, x, u, preComputation);
  const auto cloneApprox = zeroForceConstraintPtr->getLinearApproximation(t, x, u, preComputation);
  LOG(INFO) << value.isApprox(cloneValue);
  LOG(INFO) << approx.f.isApprox(cloneApprox.f);
  LOG(INFO) << approx.dfdx.isApprox(cloneApprox.dfdx);
  LOG(INFO) << approx.dfdu.isApprox(cloneApprox.dfdu);
  return 0;
}

////////////////////////////////////////// TestFrictionConeConstraint //////////////////////////////////////////////////////

int TestFrictionConeConstraint_finiteDifference() {
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  const std::shared_ptr<ocs2::legged_robot::SwitchedModelReferenceManager> referenceManagerPtr = ocs2::legged_robot::createReferenceManager(centroidalModelInfo.numThreeDofContacts);
  ocs2::PreComputation preComputation;
  const ocs2::legged_robot::FrictionConeConstraint::Config config;

  ocs2::scalar_t t = 0.0;
  ocs2::scalar_t eps = 1e-4;
  ocs2::scalar_t tol = 1e-2;  // tolerance on the Jacobian elements
  size_t N = 10000;

  for (size_t legNumber = 0; legNumber < centroidalModelInfo.numThreeDofContacts; ++legNumber) {
    ocs2::legged_robot::FrictionConeConstraint frictionConeConstraint(*referenceManagerPtr, config, legNumber, centroidalModelInfo);

    ocs2::vector_t u0 = 10.0 * ocs2::vector_t::Random(centroidalModelInfo.inputDim);
    u0(2) = 100.0;
    u0(5) = 100.0;
    u0(8) = 100.0;
    u0(11) = 100.0;
    ocs2::vector_t x0 = 0.1 * ocs2::vector_t::Random(centroidalModelInfo.stateDim);
    const auto y0 = frictionConeConstraint.getValue(t, x0, u0, preComputation)(0);
    auto quadraticApproximation = frictionConeConstraint.getQuadraticApproximation(t, x0, u0, preComputation);

    ocs2::vector_t data(N);
    ocs2::matrix_t regressor(N, 6 + 6 + 6 + 9);
    ocs2::vector_t dx = ocs2::vector_t::Zero(centroidalModelInfo.stateDim);
    ocs2::vector_t du = ocs2::vector_t::Zero(centroidalModelInfo.inputDim);
    for (size_t i = 0; i < N; i++) {
      // evaluation point
      ocs2::vector_t dEuler = eps * ocs2::vector_t::Random(3);
      ocs2::vector_t dF = eps * ocs2::vector_t::Random(3);
      dx.segment<3>(0) = dEuler;
      du.segment<3>(3 * legNumber) = dF;
      ocs2::vector_t dz(6);
      dz << dEuler, dF;
      const ocs2::matrix_t quadTerms = dz * dz.transpose();
      ocs2::vector_t quadTermsVector(6 + 6 + 9);
      size_t count = 0;
      for (size_t p = 0; p < 6; ++p) {
        for (size_t q = p; q < 6; ++q) {
          quadTermsVector(count) = quadTerms(p, q);
          if (q == p) {
            quadTermsVector(count) *= 0.5;
          }
          count++;
        }
      }

      // Scale to condition the regressor
      regressor.row(i) << dEuler.transpose() / eps, dF.transpose() / eps, quadTermsVector.transpose() / (eps * eps);
      data(i) = (frictionConeConstraint.getValue(t, x0 + dx, u0 + du, preComputation)(0) - y0);
    }
    LOG(INFO) << "data=" << data.size() << ", regressor=" << regressor.size();
    ocs2::vector_t dh_emperical = regressor.colPivHouseholderQr().solve(data);
    dh_emperical /= eps;
    dh_emperical.tail<6 + 6 + 9>() /= eps;

    ocs2::matrix_t quadTerms(6, 6);
    size_t count = 0;
    for (size_t p = 0; p < 6; ++p) {
      for (size_t q = p; q < 6; ++q) {
        quadTerms(p, q) = dh_emperical(6 + count);
        quadTerms(q, p) = quadTerms(p, q);
        count++;
      }
    }

    ocs2::vector_t dhdx_emperical = dh_emperical.head<3>();
    ocs2::vector_t dhdu_emperical = dh_emperical.segment<3>(3);
    ocs2::matrix_t ddhdxdx_emperical = quadTerms.block<3, 3>(0, 0);
    ocs2::matrix_t ddhdudx_emperical = quadTerms.block<3, 3>(3, 0);
    ocs2::matrix_t ddhdudu_emperical = quadTerms.block<3, 3>(3, 3);

    ocs2::matrix_t ddhdudu = quadraticApproximation.dfduu.front().block<3, 3>(3 * legNumber, 3 * legNumber);
    ocs2::matrix_t ddhdxdx = quadraticApproximation.dfdxx.front().block<3, 3>(0, 0);
    if((dhdx_emperical - quadraticApproximation.dfdx.block<1, 3>(0, 0).transpose()).array().abs().maxCoeff()>tol) {
      LOG(ERROR) << (dhdx_emperical - quadraticApproximation.dfdx.block<1, 3>(0, 0).transpose()).array().abs().maxCoeff();
    }
    if((dhdu_emperical - quadraticApproximation.dfdu.block<1, 3>(0, 3 * legNumber).transpose()).array().abs().maxCoeff()>tol) {
      LOG(ERROR) << (dhdu_emperical - quadraticApproximation.dfdu.block<1, 3>(0, 3 * legNumber).transpose()).array().abs().maxCoeff();
    }
    if((ddhdudu_emperical - ddhdudu).array().abs().maxCoeff()>tol) {
      LOG(ERROR) << (ddhdudu_emperical - ddhdudu).array().abs().maxCoeff();
    }
    // ddhdxdx and ddhdudx are off because of the negative definite hessian approximation
  }
  LOG(INFO) << "return";
  return 0;
}

int TestFrictionConeConstraint_gravityAligned_flatTerrain() {
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  const std::shared_ptr<ocs2::legged_robot::SwitchedModelReferenceManager> referenceManagerPtr = ocs2::legged_robot::createReferenceManager(centroidalModelInfo.numThreeDofContacts);
  ocs2::PreComputation preComputation;
  // Check friction cone for the case where the body is aligned with the terrain
  const ocs2::legged_robot::FrictionConeConstraint::Config config(0.7, 25.0, 0.0, 0.0);
  const auto mu = config.frictionCoefficient;
  const auto regularization = config.regularization;

  // evaluation point
  ocs2::scalar_t t = 0.0;
  ocs2::vector_t x = ocs2::vector_t::Random(centroidalModelInfo.stateDim);
  ocs2::vector_t u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);

  for (size_t legNumber = 0; legNumber < centroidalModelInfo.numThreeDofContacts; ++legNumber) {
    ocs2::legged_robot::FrictionConeConstraint frictionConeConstraint(*referenceManagerPtr, config, legNumber, centroidalModelInfo);

    // Local forces are equal to the body forces.
    const ocs2::vector_t F = ocs2::centroidal_model::getContactForces(u, legNumber, centroidalModelInfo);
    const auto Fx = F(0);
    const auto Fy = F(1);
    const auto Fz = F(2);
    LOG(INFO) << "frictionConeConstraint";
    auto quadraticApproximation = frictionConeConstraint.getQuadraticApproximation(t, x, u, preComputation);
    if (quadraticApproximation.f(0) != Fz * sqrt(mu * mu) - sqrt(Fx * Fx + Fy * Fy + regularization)) {
      LOG(ERROR) << "quadraticApproximation.f(0)=" << quadraticApproximation.f(0)
          << "Fz * sqrt(mu * mu) - sqrt(Fx * Fx + Fy * Fy + regularization)=" << Fz * sqrt(mu * mu) - sqrt(Fx * Fx + Fy * Fy + regularization);
    }

    // First derivative inputs
    ocs2::vector_t dhdu = ocs2::vector_t::Zero(centroidalModelInfo.inputDim);
    const auto F_norm = sqrt(Fx * Fx + Fy * Fy + regularization);
    dhdu(3 * legNumber + 0) = -Fx / F_norm;
    dhdu(3 * legNumber + 1) = -Fy / F_norm;
    dhdu(3 * legNumber + 2) = sqrt(mu * mu);
    if ((quadraticApproximation.dfdu.row(0).transpose() - dhdu).norm() > 1e-12) {
      LOG(ERROR) << (quadraticApproximation.dfdu.row(0).transpose() - dhdu).norm();
    }

    // Second derivative inputs
    ocs2::matrix_t ddhdudu = ocs2::matrix_t::Zero(centroidalModelInfo.inputDim, centroidalModelInfo.inputDim);
    const auto F_norm2 = Fx * Fx + Fy * Fy + regularization;
    const auto F_norm32 = pow(F_norm2, 1.5);
    ddhdudu(3 * legNumber + 0, 3 * legNumber + 0) = -(Fy * Fy + regularization) / F_norm32;
    ddhdudu(3 * legNumber + 0, 3 * legNumber + 1) = Fx * Fy / F_norm32;
    ddhdudu(3 * legNumber + 0, 3 * legNumber + 2) = 0.0;
    ddhdudu(3 * legNumber + 1, 3 * legNumber + 0) = Fx * Fy / F_norm32;
    ddhdudu(3 * legNumber + 1, 3 * legNumber + 1) = -(Fx * Fx + regularization) / F_norm32;
    ddhdudu(3 * legNumber + 1, 3 * legNumber + 2) = 0.0;
    ddhdudu(3 * legNumber + 2, 3 * legNumber + 0) = 0.0;
    ddhdudu(3 * legNumber + 2, 3 * legNumber + 1) = 0.0;
    ddhdudu(3 * legNumber + 2, 3 * legNumber + 2) = 0.0;
    if ((quadraticApproximation.dfduu.front() - ddhdudu).norm() > 1e-12) {
      LOG(ERROR) << (quadraticApproximation.dfduu.front() - ddhdudu).norm();
    }
  }
  LOG(INFO) << "return";
  return 0;
}

int TestFrictionConeConstraint_negativeDefinite() {
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  const std::shared_ptr<ocs2::legged_robot::SwitchedModelReferenceManager> referenceManagerPtr = ocs2::legged_robot::createReferenceManager(centroidalModelInfo.numThreeDofContacts);
  ocs2::PreComputation preComputation;

  const ocs2::legged_robot::FrictionConeConstraint::Config config;
  // evaluation point
  ocs2::scalar_t t = 0.0;
  ocs2::vector_t x = ocs2::vector_t::Random(centroidalModelInfo.stateDim);
  ocs2::vector_t u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);
  u(2) = 100.0;
  u(5) = 100.0;
  u(8) = 100.0;
  u(11) = 100.0;

  for (size_t legNumber = 0; legNumber < centroidalModelInfo.numThreeDofContacts; ++legNumber) {
    ocs2::legged_robot::FrictionConeConstraint frictionConeConstraint(*referenceManagerPtr, config, legNumber, centroidalModelInfo);

    const auto quadraticApproximation = frictionConeConstraint.getQuadraticApproximation(t, x, u, preComputation);
    if (ocs2::LinearAlgebra::symmetricEigenvalues(quadraticApproximation.dfdxx.front()).maxCoeff() > 0.0) {
      LOG(ERROR) << ocs2::LinearAlgebra::symmetricEigenvalues(quadraticApproximation.dfdxx.front()).maxCoeff();
    }
    if (ocs2::LinearAlgebra::symmetricEigenvalues(quadraticApproximation.dfduu.front()).maxCoeff() > 0.0) {
      LOG(ERROR) << ocs2::LinearAlgebra::symmetricEigenvalues(quadraticApproximation.dfduu.front()).maxCoeff();
    }
  }
  return 0;
}

//////////////////////// testEndEffectorLinearConstraint //////////////////////////////////////
int testEndEffectorLinearConstraint_testValue() {
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  ocs2::PreComputation preComputation;
  std::unique_ptr<ocs2::CentroidalModelPinocchioMapping> pinocchioMappingPtr;
  std::unique_ptr<ocs2::CentroidalModelPinocchioMappingCppAd> pinocchioMappingAdPtr;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> eeKinematicsPtr;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> eeKinematicsAdPtr;
  ocs2::legged_robot::EndEffectorLinearConstraint::Config config;
  ocs2::vector_t x, u;
  const ocs2::legged_robot::ModelSettings modelSettings;  // default constructor just to get contactNames3DoF

  pinocchioMappingPtr.reset(new ocs2::CentroidalModelPinocchioMapping(centroidalModelInfo));
  pinocchioMappingAdPtr.reset(new ocs2::CentroidalModelPinocchioMappingCppAd(centroidalModelInfo.toCppAd()));

  eeKinematicsPtr.reset(
      new ocs2::PinocchioEndEffectorKinematics(*pinocchioInterfacePtr, *pinocchioMappingPtr, {modelSettings.contactNames3DoF[0]}));

  auto velocityUpdateCallback = [&](ocs2::ad_vector_t state, ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterfaceAd) {
    const ocs2::ad_vector_t& q = state.tail(centroidalModelInfo.generalizedCoordinatesNum);
    ocs2::updateCentroidalDynamics(pinocchioInterfaceAd, centroidalModelInfo.toCppAd(), q);
  };
  eeKinematicsAdPtr.reset(new ocs2::PinocchioEndEffectorKinematicsCppAd(
      *pinocchioInterfacePtr, *pinocchioMappingAdPtr, {modelSettings.contactNames3DoF[0]}, centroidalModelInfo.stateDim,
      centroidalModelInfo.inputDim, velocityUpdateCallback, "EEVel", "/tmp/ocs2", true, true));

  x.resize(centroidalModelInfo.stateDim);
  x(0) = 0.0;  // vcom_x
  x(1) = 0.0;  // vcom_y
  x(2) = 0.0;  // vcom_z
  x(3) = 0.0;  // L_x / robotMass
  x(4) = 0.0;  // L_y / robotMass
  x(5) = 0.0;  // L_z / robotMass

  x(6) = 0.0;   // p_base_x
  x(7) = 0.0;   // p_base_y
  x(8) = 0.57;  // p_base_z
  x(9) = 0.0;   // theta_base_z
  x(10) = 0.0;  // theta_base_y
  x(11) = 0.0;  // theta_base_x

  x(12) = -0.25;  // LF_HAA
  x(13) = 0.6;    // LF_HFE
  x(14) = -0.85;  // LF_KFE
  x(15) = -0.25;  // LH_HAA
  x(16) = -0.6;   // LH_HFE
  x(17) = 0.85;   // LH_KFE
  x(18) = 0.25;   // RF_HAA
  x(19) = 0.6;    // RF_HFE
  x(20) = -0.85;  // RF_KFE
  x(21) = 0.25;   // RH_HAA
  x(22) = -0.6;   // RH_HFE
  x(23) = 0.85;   // RH_KFE

  u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);

  config.b = ocs2::vector_t::Random(3);
  config.Ax = ocs2::matrix_t::Random(3, 3);
  config.Av = ocs2::matrix_t::Random(3, 3);


  auto eeVelConstraintPtr = std::make_unique<ocs2::legged_robot::EndEffectorLinearConstraint>(*eeKinematicsPtr, 3);
  eeVelConstraintPtr->configure(config);
  auto eeVelConstraintAdPtr = std::make_unique<ocs2::legged_robot::EndEffectorLinearConstraint>(*eeKinematicsAdPtr, 3);
  eeVelConstraintAdPtr->configure(config);

  dynamic_cast<ocs2::PinocchioEndEffectorKinematics&>(eeVelConstraintPtr->getEndEffectorKinematics()).setPinocchioInterface(*pinocchioInterfacePtr);
  pinocchioMappingPtr->setPinocchioInterface(*pinocchioInterfacePtr);

  const auto& model = pinocchioInterfacePtr->getModel();
  auto& data = pinocchioInterfacePtr->getData();

  const auto q = pinocchioMappingPtr->getPinocchioJointPosition(x);
  ocs2::updateCentroidalDynamics(*pinocchioInterfacePtr, centroidalModelInfo, q);
  const auto v = pinocchioMappingPtr->getPinocchioJointVelocity(x, u);

  // For getPosition() & getVelocity() of PinocchioEndEffectorKinematics
  pinocchio::forwardKinematics(model, data, q, v);
  pinocchio::updateFramePlacements(model, data);

  const auto value = eeVelConstraintPtr->getValue(0.0, x, u, preComputation);
  const auto valueAd = eeVelConstraintAdPtr->getValue(0.0, x, u, preComputation);
  LOG(INFO) << "value=" << value.transpose() << ", valueAd=" << valueAd.transpose();
  // EXPECT_TRUE(value.isApprox(valueAd));

  return 0;
}

int testEndEffectorLinearConstraint_testLinearApproximation() {
  const ocs2::CentroidalModelType centroidalModelType = ocs2::CentroidalModelType::SingleRigidBodyDynamics;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr = ocs2::legged_robot::createAnymalPinocchioInterface();
  const ocs2::CentroidalModelInfo centroidalModelInfo = ocs2::legged_robot::createAnymalCentroidalModelInfo(*pinocchioInterfacePtr, centroidalModelType);
  ocs2::PreComputation preComputation;
  std::unique_ptr<ocs2::CentroidalModelPinocchioMapping> pinocchioMappingPtr;
  std::unique_ptr<ocs2::CentroidalModelPinocchioMappingCppAd> pinocchioMappingAdPtr;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> eeKinematicsPtr;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> eeKinematicsAdPtr;
  ocs2::legged_robot::EndEffectorLinearConstraint::Config config;
  ocs2::vector_t x, u;
  const ocs2::legged_robot::ModelSettings modelSettings;  // default constructor just to get contactNames3DoF

  pinocchioMappingPtr.reset(new ocs2::CentroidalModelPinocchioMapping(centroidalModelInfo));
  pinocchioMappingAdPtr.reset(new ocs2::CentroidalModelPinocchioMappingCppAd(centroidalModelInfo.toCppAd()));

  eeKinematicsPtr.reset(
      new ocs2::PinocchioEndEffectorKinematics(*pinocchioInterfacePtr, *pinocchioMappingPtr, {modelSettings.contactNames3DoF[0]}));

  auto velocityUpdateCallback = [&](ocs2::ad_vector_t state, ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterfaceAd) {
    const ocs2::ad_vector_t& q = state.tail(centroidalModelInfo.generalizedCoordinatesNum);
    ocs2::updateCentroidalDynamics(pinocchioInterfaceAd, centroidalModelInfo.toCppAd(), q);
  };
  eeKinematicsAdPtr.reset(new ocs2::PinocchioEndEffectorKinematicsCppAd(
      *pinocchioInterfacePtr, *pinocchioMappingAdPtr, {modelSettings.contactNames3DoF[0]}, centroidalModelInfo.stateDim,
      centroidalModelInfo.inputDim, velocityUpdateCallback, "EEVel", "/tmp/ocs2", true, true));

  x.resize(centroidalModelInfo.stateDim);
  x(0) = 0.0;  // vcom_x
  x(1) = 0.0;  // vcom_y
  x(2) = 0.0;  // vcom_z
  x(3) = 0.0;  // L_x / robotMass
  x(4) = 0.0;  // L_y / robotMass
  x(5) = 0.0;  // L_z / robotMass

  x(6) = 0.0;   // p_base_x
  x(7) = 0.0;   // p_base_y
  x(8) = 0.57;  // p_base_z
  x(9) = 0.0;   // theta_base_z
  x(10) = 0.0;  // theta_base_y
  x(11) = 0.0;  // theta_base_x

  x(12) = -0.25;  // LF_HAA
  x(13) = 0.6;    // LF_HFE
  x(14) = -0.85;  // LF_KFE
  x(15) = -0.25;  // LH_HAA
  x(16) = -0.6;   // LH_HFE
  x(17) = 0.85;   // LH_KFE
  x(18) = 0.25;   // RF_HAA
  x(19) = 0.6;    // RF_HFE
  x(20) = -0.85;  // RF_KFE
  x(21) = 0.25;   // RH_HAA
  x(22) = -0.6;   // RH_HFE
  x(23) = 0.85;   // RH_KFE

  u = ocs2::vector_t::Random(centroidalModelInfo.inputDim);

  config.b = ocs2::vector_t::Random(3);
  config.Ax = ocs2::matrix_t::Random(3, 3);
  config.Av = ocs2::matrix_t::Random(3, 3);

  auto eeVelConstraintPtr = std::make_unique<ocs2::legged_robot::EndEffectorLinearConstraint>(*eeKinematicsPtr, 3);
  eeVelConstraintPtr->configure(config);
  auto eeVelConstraintAdPtr = std::make_unique<ocs2::legged_robot::EndEffectorLinearConstraint>(*eeKinematicsAdPtr, 3);
  eeVelConstraintAdPtr->configure(config);

  dynamic_cast<ocs2::PinocchioEndEffectorKinematics&>(eeVelConstraintPtr->getEndEffectorKinematics()).setPinocchioInterface(*pinocchioInterfacePtr);
  pinocchioMappingPtr->setPinocchioInterface(*pinocchioInterfacePtr);

  const auto& model = pinocchioInterfacePtr->getModel();
  auto& data = pinocchioInterfacePtr->getData();

  // PinocchioInterface update for the analytical EndEffectorVelocityConstraint
  const auto q = pinocchioMappingPtr->getPinocchioJointPosition(x);
  updateCentroidalDynamics(*pinocchioInterfacePtr, centroidalModelInfo, q);
  const auto v = pinocchioMappingPtr->getPinocchioJointVelocity(x, u);
  const auto a = ocs2::vector_t::Zero(q.size());

  // For getPositionLinearApproximation of PinocchioEndEffectorKinematics
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::computeJointJacobians(model, data);
  // For getVelocityLinearApproximation of PinocchioEndEffectorKinematics
  pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);
  // For getOcs2Jacobian of CentroidalModelPinocchioMapping
  ocs2::updateCentroidalDynamicsDerivatives(*pinocchioInterfacePtr, centroidalModelInfo, q, v);

  const auto linApprox = eeVelConstraintPtr->getLinearApproximation(0.0, x, u, preComputation);
  const auto linApproxAd = eeVelConstraintAdPtr->getLinearApproximation(0.0, x, u, preComputation);

  LOG(INFO) << "linApprox.f=" << linApprox.f.transpose() << ", linApproxAd.f=" << linApproxAd.f.transpose();
  LOG(INFO) << "linApprox.dfdx=" << linApprox.dfdx.transpose() << ", linApproxAd.dfdx=" << linApproxAd.dfdx.transpose();
  LOG(INFO) << "linApprox.dfdu=" << linApprox.dfdu.transpose() << ", linApproxAd.dfdu=" << linApproxAd.dfdu.transpose();

  return 0;
}

int main() {
  LOG(INFO) << "LeggedRobotDemo start";
  // TestZeroForceConstraintEvaluate();
  // TestZeroForceConstraintClone();
  // TestFrictionConeConstraint_finiteDifference();
  // TestFrictionConeConstraint_gravityAligned_flatTerrain();
  // TestFrictionConeConstraint_negativeDefinite();
  // testEndEffectorLinearConstraint_testValue();
  testEndEffectorLinearConstraint_testLinearApproximation();
  LOG(INFO) << "LeggedRobotDemo End";
  return 0;
}