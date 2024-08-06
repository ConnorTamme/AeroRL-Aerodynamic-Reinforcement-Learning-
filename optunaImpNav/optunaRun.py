from ddqn_agent import DDQN_Agent
import optuna
from optuna.trial import TrialState

class test:
    def __init__(self):
        self.file = open("optunaLog.txt", "w")
        self.lrs = [0.000585, 0.000054, 0.000165, 0.00018]
        self.curLr=0

    def objective(self, trial):
        #test lr and decay rate only
        # Generate the optimizers.
        #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) #adam is fine
        optimizer_name = "Adam"
        #lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        lr = self.lrs[self.curLr]
        self.curLr += 1
       # eps_decay = trial.suggest_int("eps_decay", 60000, 75000)
        eps_decay = 70000
        #gamma = trial.suggest_float("gamma", 0.1, 0.9)#0.8 is fine
        gamma = 0.8
        ddqn_agent = DDQN_Agent(eps_decay, gamma, lr, optimizer_name, useDepth=False)
        average = ddqn_agent.train()
        log_entry = f"Optimizer : {optimizer_name} , Learning Rate : {lr} , Eps_Decay : {eps_decay} , Gamma : {gamma} , Average : {average}\n"

        self.file.write(log_entry)
        self.file.flush()
        return average

if __name__ == "__main__":
    t = test()
    study = optuna.create_study(direction="maximize")
    study.optimize(t.objective, n_trials=4)

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
