#include <cmath>
#include <glog/logging.h>
#include <ocs2_double_integrator/DoubleIntegratorInterface.h>
#include <ocs2_double_integrator/package_path.h>

#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>

class DoubleIntegrator {
public:
  DoubleIntegrator() {
    const bool verbose = true;
    const std::string taskFile = ocs2::double_integrator::getPath() + "/config/mpc/task.info";
    const std::string libFolder = ocs2::double_integrator::getPath() + "/auto_generated";
    doubleIntegratorInterfacePtr.reset(new ocs2::double_integrator::DoubleIntegratorInterface(taskFile, libFolder, verbose));

    initState = doubleIntegratorInterfacePtr->getInitialState();
    goalState = doubleIntegratorInterfacePtr->getInitialTarget();
    LOG(INFO) << "initState = " << initState.transpose() << ", goalState = " << goalState.transpose();

    // initialize reference
    ocs2::TargetTrajectories targetTrajectories({initTime}, {goalState}, {ocs2::vector_t::Zero(ocs2::double_integrator::INPUT_DIM)});
    LOG(INFO) << "targetTrajectories:\n" << targetTrajectories;
    doubleIntegratorInterfacePtr->getReferenceManagerPtr()->setTargetTrajectories(std::move(targetTrajectories));
  }

  std::unique_ptr<ocs2::GaussNewtonDDP_MPC> getMpc(bool warmStart) {
    auto& interface = *doubleIntegratorInterfacePtr;
    auto mpcSettings = interface.mpcSettings();
    auto ddpSettings = interface.ddpSettings();
    if (!warmStart) {
      mpcSettings.coldStart_ = true;
      ddpSettings.maxNumIterations_ = 5;
    }

    auto mpcPtr = std::make_unique<ocs2::GaussNewtonDDP_MPC>(std::move(mpcSettings), std::move(ddpSettings), interface.getRollout(),
                                                       interface.getOptimalControlProblem(), interface.getInitializer());
    mpcPtr->getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());

    return mpcPtr;
  }

  const ocs2::scalar_t tolerance = 2e-2;
  const ocs2::scalar_t f_mpc = 10.0;
  const ocs2::scalar_t initTime = 1234.5;  // start from a random time
  const ocs2::scalar_t finalTime = initTime + 5.0;

  ocs2::vector_t initState;
  ocs2::vector_t goalState;
  std::unique_ptr<ocs2::double_integrator::DoubleIntegratorInterface> doubleIntegratorInterfacePtr;
};

int synchronousTracking() {
  DoubleIntegrator di;
  auto mpcPtr = di.getMpc(true);
  ocs2::MPC_MRT_Interface mpcInterface(*mpcPtr);

  ocs2::SystemObservation observation;
  observation.time = di.initTime;
  observation.state = di.initState;
  observation.input.setZero(ocs2::double_integrator::INPUT_DIM);
  mpcInterface.setCurrentObservation(observation);

  // run MPC for N iterations
  auto time = observation.time;
  while (time < di.finalTime+1e-8) {
    // run MPC
    mpcInterface.advanceMpc();
    time += 1.0 / di.f_mpc;

    if (mpcInterface.initialPolicyReceived()) {
      size_t mode;
      ocs2::vector_t optimalState, optimalInput;

      mpcInterface.updatePolicy();
      mpcInterface.evaluatePolicy(time, ocs2::vector_t::Zero(ocs2::double_integrator::STATE_DIM), optimalState, optimalInput, mode);

      // use optimal state for the next observation:
      observation.time = time;
      observation.state = optimalState;
      observation.input.setZero(ocs2::double_integrator::INPUT_DIM);
      mpcInterface.setCurrentObservation(observation);
    }
  }
  LOG(INFO) << "end state=" << observation.state.transpose();

  return 0;
}

int main() {
  LOG(INFO) << "DoubleIntegratorDemo";
  synchronousTracking();

  return 0;
}