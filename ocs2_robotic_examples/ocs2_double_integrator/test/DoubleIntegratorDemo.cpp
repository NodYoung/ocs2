#include <cmath>
#include <glog/logging.h>
#include <ocs2_double_integrator/DoubleIntegratorInterface.h>
#include <ocs2_double_integrator/package_path.h>

#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>


int DoubleIntegratorIntegrationTestSynchronousTracking(bool warm_start=true) {
  const std::string task_file = ocs2::double_integrator::getPath() + "/config/mpc/task.info";
  const std::string lib_folder = ocs2::double_integrator::getPath() + "/auto_generated";
  ocs2::double_integrator::DoubleIntegratorInterface double_integrator_interface(task_file, lib_folder, true);
  LOG(INFO) << "init_state = " << double_integrator_interface.getInitialState().transpose() << ", goal_state = " << double_integrator_interface.getInitialTarget().transpose();

  const ocs2::scalar_t init_time(1234.5), final_time(1234.5+5.0);  // start from a random time
  ocs2::TargetTrajectories target_trajectories({init_time}, {double_integrator_interface.getInitialTarget()}, {ocs2::vector_t::Zero(ocs2::double_integrator::INPUT_DIM)}); // initialize reference
  LOG(INFO) << "target_trajectories:\n" << target_trajectories;
  double_integrator_interface.getReferenceManagerPtr()->setTargetTrajectories(std::move(target_trajectories));

  if (!warm_start) {
    double_integrator_interface.mpcSettings().coldStart_ = true;
    double_integrator_interface.ddpSettings().maxNumIterations_ = 5;
  }
  ocs2::GaussNewtonDDP_MPC mpc(double_integrator_interface.mpcSettings(), double_integrator_interface.ddpSettings(),
      double_integrator_interface.getRollout(), double_integrator_interface.getOptimalControlProblem(), double_integrator_interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(double_integrator_interface.getReferenceManagerPtr());
  ocs2::MPC_MRT_Interface mpc_interface(mpc);

  ocs2::SystemObservation observation;
  observation.time = init_time;
  observation.state = double_integrator_interface.getInitialState();
  observation.input.setZero(ocs2::double_integrator::INPUT_DIM);
  mpc_interface.setCurrentObservation(observation);

  // run MPC for N iterations
  const ocs2::scalar_t f_mpc = 10.0;
  ocs2::scalar_t time = observation.time;
  while (time < final_time+1e-8) {  // run MPC
    mpc_interface.advanceMpc();
    time += 1.0 / f_mpc;

    if (mpc_interface.initialPolicyReceived()) {
      size_t mode;
      ocs2::vector_t optimal_state, optimal_input;
      mpc_interface.updatePolicy();
      mpc_interface.evaluatePolicy(time, ocs2::vector_t::Zero(ocs2::double_integrator::STATE_DIM), optimal_state, optimal_input, mode);
      LOG(INFO) << "time=" << time << ", optimal_state=" << optimal_state.transpose() << ", optimal_input=" << optimal_input.transpose();
      // use optimal state for the next observation:
      observation.time = time;
      observation.state = optimal_state;
      observation.input.setZero(ocs2::double_integrator::INPUT_DIM);
      mpc_interface.setCurrentObservation(observation);
    }
  }
  LOG(INFO) << "end state=" << observation.state.transpose();

  return 0;
}

int main() {
  LOG(INFO) << "DoubleIntegratorDemo start";
  DoubleIntegratorIntegrationTestSynchronousTracking();
  LOG(INFO) << "DoubleIntegratorDemo end";
  return 0;
}