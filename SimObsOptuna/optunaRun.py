from ddqn_agent import DDQN_Agent
import optuna
from optuna.trial import TrialState

file = open("optunaLog.txt", "w")

def objective(trial):
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    eps_decay = trial.suggest_int("eps_decay", 25000, 50000)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    ddqn_agent = DDQN_Agent(eps_decay, gamma, lr, optimizer_name, useDepth=False)
    average = ddqn_agent.train()
    log_entry = f"Optimizer : {optimizer_name} , Learning Rate : {lr} , Eps_Decay : {eps_decay} , Gamma : {gamma} , Average : {average}\n"

    file.write(log_entry)
    return average

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
