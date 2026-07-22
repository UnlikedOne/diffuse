# Self-hosting a private network

You do not have to use the public network. A team, a lab, or a home setup can run
its own private Diffuse network, where discovery, routing, and relaying all stay
among machines you control.

## The idea

Every node and client finds the network through a **sentinel**: a well-known
address used for bootstrap discovery and, when needed, as an encrypted relay. Run
your own sentinel and point everything at it with `--bootstrap`, and you have a
private network that never touches the public one.

## Step 1: start the first node

Pick one machine to be the entry point. Its gossip address is your sentinel.

```bash
diffuse host \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --listen 0.0.0.0:9440 \
  --public-addr YOUR_SERVER_IP:9440 \
  --bootstrap http://YOUR_SERVER_IP:9440
```

`--public-addr` is the address other machines will use to reach it. Bootstrapping
it to itself makes it a self-contained network of one, ready for others to join.

## Step 2: add more nodes

On each additional machine, join through the sentinel:

```bash
diffuse host \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --bootstrap http://YOUR_SERVER_IP:9440
```

Each node profiles itself and takes the slice the network needs most, so add
machines and coverage fills in. Wait for one to report its slice before starting
the next, so assignments fill gaps in order.

## Step 3: use it

Point clients at your sentinel instead of the public one:

```bash
diffuse chat --bootstrap http://YOUR_SERVER_IP:9440
diffuse query --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello" \
  --bootstrap http://YOUR_SERVER_IP:9440
diffuse serve --model Qwen/Qwen2.5-0.5B-Instruct --bootstrap http://YOUR_SERVER_IP:9440
```

## Nodes behind NAT

Machines on home networks that cannot accept inbound connections will serve
through the sentinel [relay](/concepts/nat-relay) automatically, as long as the
sentinel has a reachable public address. This is why the first node should be on a
machine others can reach.

## Keeping it running

Run each node detached and, for a real deployment, under a process supervisor so
it restarts after a reboot:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct \
  --bootstrap http://YOUR_SERVER_IP:9440 > ~/.diffuse/host.log 2>&1 &
```

## Honest limits

- The sentinel is a **single point of failure** for relayed nodes today. If it
  goes down, nodes attached to it become unreachable and do not reattach
  elsewhere automatically.
- There is **no admission control**, so on an open sentinel anyone who knows the
  address can join. Keep the address private, or run it on a network only your
  machines can reach.

See the full list on the [limitations](/limitations) page.
