#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>
#include <glog/logging.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/LoadStdVectorOfPair.h>
#include <ocs2_robotic_assets/package_path.h>
#include <ocs2_self_collision/SelfCollision.h>
#include <ocs2_self_collision/SelfCollisionCppAd.h>

#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_self_collision/PinocchioGeometryInterface.h>

#include "ocs2_mobile_manipulator/FactoryFunctions.h"
#include "ocs2_mobile_manipulator/ManipulatorModelInfo.h"
#include "ocs2_mobile_manipulator/MobileManipulatorInterface.h"
#include "ocs2_mobile_manipulator/MobileManipulatorPinocchioMapping.h"
#include "ocs2_mobile_manipulator/MobileManipulatorPreComputation.h"
#include "ocs2_mobile_manipulator/constraint/EndEffectorConstraint.h"
#include "ocs2_mobile_manipulator/package_path.h"


/////////////////////////////////////////////// TestSelfCollision /////////////////////////////////////////////////////


class TestSelfCollision {
public:
  TestSelfCollision() : pinocchioInterface(createMobileManipulatorPinocchioInterface()), geometryInterface(pinocchioInterface, collisionPairs) {}

  void computeValue(ocs2::PinocchioInterface& pinocchioInterface, const ocs2::vector_t q) {
    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();
    pinocchio::forwardKinematics(model, data, q);
  }

  void computeLinearApproximation(ocs2::PinocchioInterface& pinocchioInterface, const ocs2::vector_t q) {
    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();
    pinocchio::computeJointJacobians(model, data, q);  // also computes forwardKinematics
    pinocchio::updateGlobalPlacements(model, data);
  }

  // initial joint configuration
  const ocs2::vector_t jointPositon = (ocs2::vector_t(9) << 1.0, 1.0, 0.5, 2.5, -1.0, 1.5, 0.0, 1.0, 0.0).finished();
  const std::vector<std::pair<size_t, size_t>> collisionPairs = {{1, 4}, {1, 6}, {1, 9}};

  const std::string libraryFolder = ocs2::mobile_manipulator::getPath() + "/auto_generated";
  const ocs2::scalar_t minDistance = 0.1;

  ocs2::PinocchioInterface pinocchioInterface;
  ocs2::PinocchioGeometryInterface geometryInterface;

protected:
  ocs2::PinocchioInterface createMobileManipulatorPinocchioInterface() {
    const std::string urdfPath = ocs2::robotic_assets::getPath() + "/resources/mobile_manipulator/mabi_mobile/urdf/mabi_mobile.urdf";
    const std::string taskFile = ocs2::mobile_manipulator::getPath() + "/config/mabi_mobile/task.info";

    // read manipulator type
    ocs2::mobile_manipulator::ManipulatorModelType modelType = ocs2::mobile_manipulator::loadManipulatorType(taskFile, "model_information.manipulatorModelType");
    // read the joints to make fixed
    std::vector<std::string> removeJointNames;
    ocs2::loadData::loadStdVector<std::string>(taskFile, "model_information.removeJoints", removeJointNames, false);
    // initialize pinocchio interface
    return createPinocchioInterface(urdfPath, modelType, removeJointNames);
  }
};

int SelfCollisionRandomJointPositions() {
  TestSelfCollision tsc;
  ocs2::SelfCollision selfCollision(tsc.geometryInterface, tsc.minDistance);
  ocs2::SelfCollisionCppAd selfCollisionCppAd(tsc.pinocchioInterface, tsc.geometryInterface, tsc.minDistance, "testSelfCollision", tsc.libraryFolder, true, false);

  for (int i = 0; i < 10; i++) {
    ocs2::vector_t q = ocs2::vector_t::Random(9);
    tsc.computeLinearApproximation(tsc.pinocchioInterface, q);

    ocs2::vector_t d1, d2;
    ocs2::matrix_t Jd1, Jd2;

    std::tie(d1, Jd1) = selfCollision.getLinearApproximation(tsc.pinocchioInterface);
    std::tie(d2, Jd2) = selfCollisionCppAd.getLinearApproximation(tsc.pinocchioInterface, q);

    if (!d1.isApprox(d2)) {
      std::cerr << "[d1]: " << d1.transpose() << '\n';
      std::cerr << "[d2]: " << d2.transpose() << '\n';
    }
    if (!Jd1.isApprox(Jd2)) {
      std::cerr << "[Jd1]:\n" << Jd1 << '\n';
      std::cerr << "[Jd2]:\n" << Jd2 << '\n';
    }
  }
}

//////////////////////////////// testEndEffectorConstraint ////////////////////////////////////////////////

ocs2::PinocchioInterface createMobileManipulatorPinocchioInterface() {
  // files
  const std::string urdfPath = ocs2::robotic_assets::getPath() + "/resources/mobile_manipulator/mabi_mobile/urdf/mabi_mobile.urdf";
  const std::string taskFile = ocs2::mobile_manipulator::getPath() + "/config/mabi_mobile/task.info";
  // read manipulator type
  ocs2::mobile_manipulator::ManipulatorModelType modelType = ocs2::mobile_manipulator::loadManipulatorType(taskFile, "model_information.manipulatorModelType");
  // read the joints to make fixed
  std::vector<std::string> removeJointNames;
  ocs2::loadData::loadStdVector<std::string>(taskFile, "model_information.removeJoints", removeJointNames, false);
  // initialize pinocchio interface
  return createPinocchioInterface(urdfPath, modelType, removeJointNames);
}

ocs2::mobile_manipulator::ManipulatorModelInfo loadManipulatorModelInfo(ocs2::PinocchioInterface pinocchioInterface) {
  // LOG(INFO) << "loadManipulatorModelInfo";
  // files
  const std::string taskFile = ocs2::mobile_manipulator::getPath() + "/config/mabi_mobile/task.info";
  // read the task file
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  // resolve meta-information about the model
  // read manipulator type
  ocs2::mobile_manipulator::ManipulatorModelType modelType = ocs2::mobile_manipulator::loadManipulatorType(taskFile, "model_information.manipulatorModelType");
  // read the frame names
  std::string baseFrame, eeFrame;
  ocs2::loadData::loadPtreeValue<std::string>(pt, baseFrame, "model_information.baseFrame", true);
  ocs2::loadData::loadPtreeValue<std::string>(pt, eeFrame, "model_information.eeFrame", true);
  // return model
  return ocs2::mobile_manipulator::createManipulatorModelInfo(pinocchioInterface, modelType, baseFrame, eeFrame);
}

class testEndEffectorConstraint {
public:
  using quaternion_t = ocs2::mobile_manipulator::EndEffectorConstraint::quaternion_t;
  using vector3_t = ocs2::mobile_manipulator::EndEffectorConstraint::vector3_t;

  testEndEffectorConstraint(ocs2::PinocchioInterface pinocchio_interface, ocs2::mobile_manipulator::ManipulatorModelInfo model_info) 
        : pinocchioInterface(pinocchio_interface), pinocchioMapping(model_info) {
    modelInfo = model_info;
    // initialize reference managers
    const ocs2::vector_t positionOrientation = (ocs2::vector_t(7) << vector3_t::Zero(), quaternion_t(1, 0, 0, 0).coeffs()).finished();
    referenceManagerPtr.reset(new ocs2::ReferenceManager(ocs2::TargetTrajectories({0.0}, {positionOrientation})));

    // initialize kinematics
    eeKinematicsPtr.reset(new ocs2::PinocchioEndEffectorKinematics(pinocchioInterface, pinocchioMapping, {modelInfo.eeFrame}));
    preComputationPtr.reset(new ocs2::mobile_manipulator::MobileManipulatorPreComputation(pinocchioInterface, modelInfo));

    x.resize(modelInfo.stateDim);
    x << 1.0, 1.0, 0.5, 2.5, -1.0, 1.5, 0.0, 1.0, 0.0;
  }

  ocs2::vector_t x;
  ocs2::PinocchioInterface pinocchioInterface;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> eeKinematicsPtr;
  std::unique_ptr<ocs2::mobile_manipulator::MobileManipulatorPreComputation> preComputationPtr;
  std::shared_ptr<ocs2::ReferenceManager> referenceManagerPtr;
  ocs2::mobile_manipulator::MobileManipulatorPinocchioMapping pinocchioMapping;
  ocs2::mobile_manipulator::ManipulatorModelInfo modelInfo;
};

int EndEffectorConstraintEvaluation() {
  ocs2::PinocchioInterface pinocchio_interface = createMobileManipulatorPinocchioInterface();
  ocs2::mobile_manipulator::ManipulatorModelInfo model_info = loadManipulatorModelInfo(pinocchio_interface);
  testEndEffectorConstraint teec(pinocchio_interface, model_info);
  ocs2::mobile_manipulator::EndEffectorConstraint eeConstraint(*(teec.eeKinematicsPtr), *(teec.referenceManagerPtr));

  auto& pinocchioInterface = teec.preComputationPtr->getPinocchioInterface();
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();
  const auto q = teec.pinocchioMapping.getPinocchioJointPosition(teec.x);
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::computeJointJacobians(model, data);

  std::cerr << "constraint:\n" << eeConstraint.getValue(0.0, teec.x, *(teec.preComputationPtr)) << '\n';
  std::cerr << "approximation:\n" << eeConstraint.getLinearApproximation(0.0, teec.x, *(teec.preComputationPtr));
  return 0;
}

//////////////////////////////////////////////////  //////////////////////////////////////////////////////////
using vector3_t = Eigen::Matrix<ocs2::scalar_t, 3, 1>;
using quaternion_t = Eigen::Quaternion<ocs2::scalar_t, Eigen::DontAlign>;

class DummyMobileManipulatorParametersTests {
public:
  // Constants
  static constexpr ocs2::scalar_t tolerance = 1e-2;
  static constexpr ocs2::scalar_t f_mpc = 10.0;
  static constexpr ocs2::scalar_t initTime = 1234.5; // start from a random time
  static constexpr ocs2::scalar_t finalTime = initTime + 10.0;

  /**
   * @note: We separate `initialize()` and `getMpc()` since one can obtain
   * multiple MPC instances from same interface.
   */
  bool initialize(const std::string& taskFile, const std::string& libFolder, const std::string& urdfFile, const vector3_t& goal_position, const quaternion_t& goal_orientation) {
    // create mpc interface
    mobileManipulatorInterfacePtr.reset(new ocs2::mobile_manipulator::MobileManipulatorInterface(taskFile, libFolder, urdfFile));
    // obtain robot model info
    modelInfo = mobileManipulatorInterfacePtr->getManipulatorModelInfo();

    // initialize reference
    const ocs2::vector_t goalState = (ocs2::vector_t(7) << goal_position, goal_orientation.coeffs()).finished();
    ocs2::TargetTrajectories targetTrajectories({initTime}, {goalState}, {ocs2::vector_t::Zero(modelInfo.inputDim)});
    mobileManipulatorInterfacePtr->getReferenceManagerPtr()->setTargetTrajectories(std::move(targetTrajectories));

    // initialize kinematics
    const std::string modelName = "end_effector_kinematics_dummytest";
    ocs2::mobile_manipulator::MobileManipulatorPinocchioMappingCppAd pinocchioMapping(modelInfo);
    const auto& pinocchioInterface = mobileManipulatorInterfacePtr->getPinocchioInterface();
    eeKinematicsPtr.reset(new ocs2::PinocchioEndEffectorKinematicsCppAd(
        pinocchioInterface, pinocchioMapping, {modelInfo.eeFrame},
        modelInfo.stateDim, modelInfo.inputDim, modelName));
    return true;
  }

  std::unique_ptr<ocs2::GaussNewtonDDP_MPC> getMpc() {
    auto& interface = *mobileManipulatorInterfacePtr;
    auto mpcPtr = std::make_unique<ocs2::GaussNewtonDDP_MPC>(
        interface.mpcSettings(), interface.ddpSettings(),
        interface.getRollout(), interface.getOptimalControlProblem(),
        interface.getInitializer());
    mpcPtr->getSolverPtr()->setReferenceManager(
        mobileManipulatorInterfacePtr->getReferenceManagerPtr());
    return mpcPtr;
  }

  void verifyTrackingQuality(const ocs2::vector_t& state, const std::string& taskFile, const std::string& urdfFile, const vector3_t& goal_position, const quaternion_t& goal_orientation) const {
    const vector3_t eePositionError = eeKinematicsPtr->getPosition(state).front() - goal_position;
    const vector3_t eeOrientationError = eeKinematicsPtr->getOrientationError(state, {goal_orientation}).front();
    // test report
    std::cerr << "[SUMMARY]: ------------------------------------------------------\n";
    std::cerr << getTestName(taskFile, urdfFile);
    std::cerr << "\teePositionError: " << eePositionError.transpose() << '\n';
    std::cerr << "\teeOrientationError: " << eeOrientationError.transpose() << '\n';
    std::cerr << "-----------------------------------------------------------------\n";
    // check that goal position is reached
    if (std::fabs(eePositionError.x())>tolerance) {
      LOG(ERROR) << eePositionError.x();
    }
    if (std::fabs(eePositionError.y())>tolerance) {
      LOG(ERROR) << eePositionError.y();
    }
    if (std::fabs(eePositionError.z())>tolerance) {
      LOG(ERROR) << eePositionError.z();
    }
    if (std::fabs(eeOrientationError.x())>tolerance) {
      LOG(ERROR) << eeOrientationError.x();
    }
    if (std::fabs(eeOrientationError.y())>tolerance) {
      LOG(ERROR) << eeOrientationError.y();
    }
    if (std::fabs(eeOrientationError.z())>tolerance) {
      LOG(ERROR) << eeOrientationError.z();
    }
  }

  const std::string getTestName(const std::string& taskFile, const std::string& urdfFile) const {
    std::string testName;
    testName += "DummyMobileManipulatorParametersTests Test {";
    testName += "\n\tTask: " + taskFile;
    testName += "\n\tURDF: " + urdfFile;
    testName += "\n}\n";
    return testName;
  }

  ocs2::mobile_manipulator::ManipulatorModelInfo modelInfo;
  std::unique_ptr<ocs2::mobile_manipulator::MobileManipulatorInterface> mobileManipulatorInterfacePtr;
  std::unique_ptr<ocs2::PinocchioEndEffectorKinematicsCppAd> eeKinematicsPtr;
};

// expose constants globally
constexpr ocs2::scalar_t DummyMobileManipulatorParametersTests::tolerance;
constexpr ocs2::scalar_t DummyMobileManipulatorParametersTests::f_mpc;
constexpr ocs2::scalar_t DummyMobileManipulatorParametersTests::initTime;
constexpr ocs2::scalar_t DummyMobileManipulatorParametersTests::finalTime;

int DummyMobileManipulatorsynchronousTracking(std::string taskFile, std::string libraryFolder, std::string urdfFile, vector3_t goalPosition, quaternion_t goalOrientation) {
  // Obtain mpc
  DummyMobileManipulatorParametersTests dmmpt;
  std::string task_file = ocs2::mobile_manipulator::getPath() + "/config/" + taskFile;
  std::string library_folder = ocs2::mobile_manipulator::getPath() + "/auto_generated/" + libraryFolder;
  std::string urdf_file = ocs2::robotic_assets::getPath() + "/resources/mobile_manipulator/" + urdfFile;
  vector3_t goal_position(goalPosition);
  quaternion_t goal_orientation(goalOrientation);

  dmmpt.initialize(task_file, library_folder, urdf_file, goal_position, goal_orientation);
  auto mpcPtr = dmmpt.getMpc();
  ocs2::MPC_MRT_Interface mpcInterface(*mpcPtr);

  // Set initial observation
  ocs2::SystemObservation observation;
  observation.time = dmmpt.initTime;
  observation.state = dmmpt.mobileManipulatorInterfacePtr->getInitialState();
  observation.input.setZero(dmmpt.modelInfo.inputDim);
  mpcInterface.setCurrentObservation(observation);

  // Run MPC for N iterations
  auto time = observation.time;
  ocs2::vector_t optimalState, optimalInput;
  while (time < dmmpt.finalTime) {
    // run MPC
    mpcInterface.advanceMpc();
    time += 1.0 / dmmpt.f_mpc;

    if (mpcInterface.initialPolicyReceived()) {
      size_t mode;
      

      mpcInterface.updatePolicy();
      mpcInterface.evaluatePolicy(time, ocs2::vector_t::Zero(dmmpt.modelInfo.stateDim),
                                  optimalState, optimalInput, mode);

      // use optimal state for the next observation:
      observation.time = time;
      observation.state = optimalState;
      observation.input.setZero(dmmpt.modelInfo.inputDim);
      mpcInterface.setCurrentObservation(observation);
    }
  }

  dmmpt.verifyTrackingQuality(observation.state, task_file, urdf_file, goal_position, goal_orientation);
  return 0;
}

int main() {
  LOG(INFO) << "MobileManipulatorDemo";
  // SelfCollisionRandomJointPositions();
  // EndEffectorConstraintEvaluation();
  DummyMobileManipulatorsynchronousTracking("franka/task.info", "franka", "franka/urdf/panda.urdf",
                        vector3_t(0.4, 0.1, 0.5), quaternion_t(0.33, 0.0, 0.0, 0.95));
  LOG(INFO) << "MobileManipulatorDemo End";
  return 0;
}