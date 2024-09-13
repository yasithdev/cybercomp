from cybercomp import Manager

if __name__ == "__main__":
    type_checker = Manager()


experiment_1 = Experiment(
name="<name/for/exp>",
model="name of model",
engine = "name of engine",
parameters={
"neuro/network_config": "<value>",
"neuro/network_config": "<value>",
},
hyperparameters={
"neuro/network_config": "<value>",
"neuro/network_config": "<value>",
},
observations={
"neuro/network_config": "<value>",
"neuro/network_config": "<value>",
}
)

# for experiment, we can give exp colln as a parameter,

and when we do that, the intersn of the collections observbns
should be an input parameter for the expnt.

# for the autocompletion, check model+engine

# and derive required and optional params,

# only show them in autocompletion.

[
"2024/04/03": "...
]

experiment_2 = create_new_experiment_from(experiment_1)
experiment_2.name = "<>"
experiment_2.parameters["neuro/network_config"] = "new-value"
experiment.check_existing()

experiment.observations

experiemnt.validate()
experiment.run(
hpc_recipe={}
)

# !!! [collection] - a set of experiments with common observations

# observations may be a huge list, so need not provide everytime when its

implictly discoverable

# to get experiments run with different observations

collection = create_collection(
model="name_of_model",
parameters={
"neuro/network_config": [],
},
)

# the collection experiments are pulled from the db

collection = create_collection(
model=["model1", "model2", ...],
parameters={
"neuro/network_config": [],
},
observations={
"neuro/network_config": [],
},
) -> [list of experiments]

# all experiments sharing the same observations

collection = create_collection(
observations={
"neuro/network_config": [],
},
) -> [list of experiments]

collection = [experiment_1, experiment_2]

# experiment collection

# example of experiment chaining (top-to-bottom mro)

# example 1

experiment = Experiment(
experiment_1,
experiment_2,
)

# example 2

experiment = Experiment(
[experiment_2, experiment_3, .....], #
experiment_1, #
)

#

[
exp2 -> exp1,
exp3 -> exp1,
]

# example 3

experiment = Experiment(
[experiment_2, ...collection.experiments],
experiment_1,
)

# analysis part =========================

# takes a collection as input,

# and runs some function over the observables on that

# collection

# a primitive form of experiment using a collection of experiments as input

analysis = Analysis(
collection=[],
function={

    }

)

analysis = experiment