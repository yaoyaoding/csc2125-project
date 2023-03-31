from __future__ import annotations
from typing import Optional, Set, Dict, List, Tuple, Deque
import random
from math import ceil
from collections import deque
from argparse import ArgumentParser

random.seed('a good seed')

parser = ArgumentParser('Simulate block generation in a block chain system.')
parser.add_argument('-n', type=int, default=100, help='Number of peer nodes.')
parser.add_argument('-u', type=int, default=50, help='Time unit of each simulation step, in millisecond.')
parser.add_argument('-t', type=float, default=1.0, help='The number of mining hours to simulate.')
parser.add_argument('-s', type=float, default=0.1, help='The number of blocks mined per minute.')
parser.add_argument('-l', type=int, default=1000, help='The network latency between two peer nodes in millisecond.')
parser.add_argument('-d', type=str, default='uniform', help='The distribution of compute power.')
parser.add_argument('-e', type=float, default=0.1, help='Edge density of the peer network.')


def round_with(a: int, b: int):
    return (a + b - 1) // b * b


class Configuration:
    def __init__(
            self,
            time_unit: int = 10,
            num_nodes: int = 100,
            blocks_per_minute: float = 0.1,
            power_distribution: Optional[ComputePowerDistribution] = None,
            network_latency: int = 1000,
            edge_density: float = 0.1,
            mining_hours: float = 1.0
    ):
        if power_distribution is None:
            power_distribution = UniformDistribution(num_nodes)
        self.time_unit: int = time_unit  # time unit for each step in millisecond
        self.num_nodes: int = num_nodes
        self.blocks_per_minute: float = blocks_per_minute
        self.power_distribution: ComputePowerDistribution = power_distribution
        self.network_latency: int = network_latency
        self.edge_density: float = edge_density
        self.mining_hours: float = mining_hours

        self.generate_prob: float = self.blocks_per_minute / 60 / 1000 * self.time_unit


class ComputePowerDistribution:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, idx: int) -> float:  # [0, n)
        raise NotImplementedError()

    @staticmethod
    def resolve_distribution(s: str, num_nodes: int):
        if s == 'uniform':
            return UniformDistribution(num_nodes)
        elif s.startswith('polar'): # e.g., 'polar:2:0.6'
            _, num_polars, polar_ratio = s.split(':')
            return PolarizedDistribution(num_nodes, int(num_polars), float(polar_ratio))
        else:
            raise RuntimeError('Can not recognize distribution: {}'.format(s))


class UniformDistribution(ComputePowerDistribution):
    def __call__(self, idx: int) -> float:
        return 1.0 / self.n


class PolarizedDistribution(ComputePowerDistribution):
    def __init__(self, n: int, num_polars: int = 2, polar_ratio: float = 0.6):
        super().__init__(n)
        self.num_polars = num_polars
        self.polar_ratio = polar_ratio

    def __call__(self, idx: int):
        if idx < self.num_polars:
            return self.polar_ratio / self.num_polars
        else:
            return (1 - self.polar_ratio) / (self.n - self.num_polars)


class Block:
    id_counter = 0

    def __init__(self, prev: Optional[Block], miner: int):
        Block.id_counter += 1
        self.id = Block.id_counter
        self.prev: Optional[Block] = prev
        self.depth: int = prev.depth + 1 if prev is not None else 1
        self.miner: int = miner


class BlockChain:
    chain_root = Block(None, 0)

    def __init__(self):
        self.blocks: Set[Block] = {self.chain_root}

    def add_block(self, block: Block):
        if block in self.blocks:
            return
        else:
            self.add_block(block.prev)
            self.blocks.add(block)

    def longest(self) -> Block:
        ret = None
        for node in self.blocks:
            if ret is None or ret.depth < node.depth:
                ret = node
        return ret

    def longest_chain_blocks(self) -> int:
        return self.longest().depth

    def num_blocks(self) -> int:
        return len(self.blocks)

    def num_folks(self) -> int:
        folk_ends = set(self.blocks)
        for node in self.blocks:
            if node.prev in folk_ends:
                folk_ends.remove(node.prev)
        return len(folk_ends) - 1


class Node:
    def __init__(self, config: Configuration, idx: int):
        self.idx: int = idx
        self.config: Configuration = config
        self.chain: BlockChain = BlockChain()
        self.receive_queue: Dict[Block, int] = {}

        self.generate_prob: float = self.config.generate_prob * self.config.power_distribution(idx)

    def update_received_blocks(self):
        for block in list(self.receive_queue.keys()):
            self.receive_queue[block] -= 1
            if self.receive_queue[block] == 0:
                del self.receive_queue[block]
                self.chain.add_block(block)

    def mine(self) -> Optional[Block]:
        success = random.random() < self.generate_prob
        if success:
            return Block(prev=self.chain.longest(), miner=self.idx)
        else:
            return None

class Network:
    def __init__(self, config: Configuration):
        self.config: Configuration = config
        self.nodes: List[Node] = []
        self.edges: Dict[Node, List[Node]] = {}
        self.distance: Dict[Tuple[Node, Node], int] = {}

        self.build_network()

    def simulate_loop(self):
        num_steps: int = int(self.config.mining_hours * 60 * 60 * 1000 / self.config.time_unit)
        for i in range(num_steps):
            self.step()
            if i % 100 == 0:
                print(self.status_summary(i, num_steps))

        # only transfer the mined blocks, to make sure all nodes have the same block chain
        wrap_up_steps = int(max(self.distance.values()) * self.config.network_latency / self.config.time_unit)
        for i in range(wrap_up_steps):
            self.step(enable_mining=False)

    def step(self, enable_mining=True):
        for node in self.nodes:
            node.update_received_blocks()
            if not enable_mining:
                continue
            mined_node = node.mine()
            if mined_node is not None:
                self.broadcast(mined_node)

    def broadcast(self, block: Block):
        miner_node: Node = self.nodes[block.miner]
        for node in self.nodes:
            if node is miner_node:
                continue
            transfer_latency = self.distance[(miner_node, node)] * self.config.network_latency
            node.receive_queue[block] = int(ceil(transfer_latency / self.config.time_unit))

    def build_network(self):
        n = self.config.num_nodes
        for i in range(n):
            node = Node(self.config, idx=i)
            self.nodes.append(node)
            self.edges[node] = []
        remain_edges: Set[Tuple[Node, Node]] = set()
        for i in range(n):
            for j in range(i):
                remain_edges.add((self.nodes[j], self.nodes[i]))

        # build a tree first, make sure this network is connected
        for i in range(1, n):
            j = random.randint(0, i - 1)
            ni, nj = self.nodes[i], self.nodes[j]
            self.edges[ni].append(nj)
            self.edges[nj].append(ni)
            remain_edges.remove((nj, ni))

        # add the remaining edges to satisfy the edge density
        num_expect_edges = int(n * (n - 1) // 2 * self.config.edge_density)
        num_extra_edges = max(0, num_expect_edges - (n - 1))
        extra_edges = random.sample(remain_edges, num_extra_edges)
        for u, v in extra_edges:
            self.edges[u].append(v)
            self.edges[v].append(u)

        # calculate the distance between different nodes in the peer network
        for root in self.nodes:
            dist: Dict[Node, int] = {root: 0}
            queue: Deque[Node] = deque([root])
            # bfs start from root
            while len(queue) > 0:
                u = queue.popleft()
                for v in self.edges[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for v, d in dist.items():
                self.distance[(root, v)] = d

    def status_summary(self, step, total) -> str:
        chain: BlockChain = self.nodes[0].chain
        return '[{:5}/{:5}] Nodes {}, Blocks {}/{}, Folks {}'.format(
            step, total, len(self.nodes), chain.longest_chain_blocks(), chain.num_blocks(), chain.num_folks()
        )


def main(arg_string: Optional[str]):
    if arg_string:
        args = parser.parse_args(arg_string.split())
    else:
        args = parser.parse_args()
    config = Configuration(
        time_unit=args.u,
        num_nodes=args.n,
        blocks_per_minute=args.s,
        power_distribution=ComputePowerDistribution.resolve_distribution(args.d, args.n),
        network_latency=round_with(args.l, args.u),
        edge_density=args.e,
        mining_hours=args.t
    )
    network = Network(config)
    network.simulate_loop()


if __name__ == '__main__':
    # main('-s 1')
    main('-s 1 -d polar:2:0.9')
