# Crayon integration to use Tensorboard

These are the guidelines to use the [Crayon library](https://github.com/torrvision/crayon) with Dynet.

Crayon is a python library that enables the integration of arbitrary Neural Networks frameworks with [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). Thus, this enables using the visualisation power that Tensorboard
provides without developing specific implementations to interact with the visualisation tool.
This is particularly useful to visualise relevant information of different experiments, for instance the evolution of the train/development loss.


## Requirements
- Dynet
- Crayon
- Docker

For Dynet, the installation details are detailed in the [repository README](../../../README.md). For the Crayon client, you can use `pip` directly:

    sudo pip install pycrayon

Docker should also be installed and can be done via your OS package manager, for **Ubuntu** should be:

    sudo apt-get install docker-ce
For more **Ubuntu** details check the following [page](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository). If you have a distribution with *zypper* it should look like this:

    sudo zypper in docker

## Running

### Running docker
First thing is to download the Crayon docker image and run it. You should have root access and guarantee that docker is running. To check if it's running one of the following should apply:

    systemctl status docker.service
    sudo service docker status

 If docker is not active/running, you can run it with one of the following:

    sudo systemctl start docker.service
    sudo service docker start

You may need to restart docker at some point, if so run the above with ***restart*** instead of *start*.

After guaranteeing that docker is running, it's time to pull the crayon server docker image:

    sudo docker pull alband/crayon
If everything goes as expected, you should have a new image in your system, you can check with:

    sudo docker images
Finally, running the server is as follows:

    sudo docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon

To check whether the container is running you can use:

    sudo docker ps -a
Notice that this will run in your localhost, if you want to run this in a remote server you need to configure your remote server to accept external ips or, easier, port forward with ssh ([this](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server) stack overflow question/answer has some details regarding this problem).

### Run your experiments
To illustrate the Crayon usage we will use the *rnnlm-batch* example.

The way Crayon works is by providing a wrapper to tensorboard, a server that interacts with the visualisation board. Thus, to create a connection to the server
crayon requires a CrayonClient to be created:

```
# Connect to the server
cc = CrayonClient(hostname=args.crayserver)
```

This establishes a connection to the server with name hostname, for illustration purposes we will use localhost.

After establishing the connection, the API requires a creation of one experiment:

```
#Create a new experiment
myexp = cc.create_experiment(args.expname)
```

Each experiment should have a name so that it's possible to identify it. This also allows to save and load experiments to and from the server.
Crayon supports adding histograms and scalar values, we will illustrate the usage with scalar plots:
```
errs, mb_chars = lm.BuildLMGraph(train[sid: sid + MB_SIZE])
loss += errs.scalar_value()
# Add a scalar value to the experiment for the set of data points named loss evolution
myexp.add_scalar_value("lossevolution", loss)
```

Checking tensorboard in port 8888, we can see the scalar plot for the loss and by hovering over a particular line we can see details:

![FOO experiment scalar lossevolution plot](tensorboardexample.png)

Finally, to save the experiment on the server:
```
# To save the experiment
filename = myexp.to_zip()
print("Save tensorboard experiment at {}".format(filename))
```

Note that if you want different scalar plots for different experiments you need to change the first parameter of ```myexp.add_scalar_value("loss evolution", loss)``` so that it matches the scalar plot you desire, otherwise the different experiment will be plotted over the same scalar plot.
Running the same code again but with a different experiment name (same scalar plot name) yields:

![FOO and BAR experiments scalar lossevolution plot](tensorboardexample2.png)

Another important feature is loading the experiment, for instance because training was stopped and you want to start at some saved point. To load the experiment directly from the server it just requires:
```python
# Connect to the server
cc = CrayonClient(hostname="localhost")

# Open an experiment
foo = cc.open_experiment("foo")

# Get the datas sent to the server
print(foo.get_scalar_values("lossevolution"))
```

**IMPORTANT NOTE: BE CAREFUL WITH NAMING THE SCALAR PLOTS. IF YOU WANT TO OPEN AN EXPERIMENT FROM THE SERVER, USING A SCALAR NAME WITH A SPACE WILL END IN AN ERROR.**

### Other informations
Scalars are not the only plots supported by crayon interface, histograms are also possible but to see more information you can see the full [API documentation](https://github.com/torrvision/crayon/blob/master/doc/specs.md).
