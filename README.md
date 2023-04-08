# Usage
```console
$ git clone https://github.com/yaoyaoding/csc2125-project
$ cd csc2125-project
$ python main.py
```

There are some options to control the simulation process:
```console
$ python main.py --help
usage: Simulate block generation in a block chain system. [-h] [-n N] [-u U] [-t T] [-s S] [-l L] [-d D] [-e E]

optional arguments:
  -h, --help  show this help message and exit
  -n N        Number of peer nodes.
  -u U        Time unit of each simulation step, in millisecond.
  -t T        The number of mining hours to simulate.
  -s S        The number of blocks mined per minute.
  -l L        The network latency between two peer nodes in millisecond.
  -d D        The distribution of compute power.
  -e E        Edge density of the peer network.

```
