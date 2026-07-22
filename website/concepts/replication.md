# Replication and healing

Machines on a volunteer network come and go. Diffuse assumes this and treats
redundancy and repair as normal operation rather than exceptional.

## Replicas

A slice can be held by more than one node. Each additional holder is a **replica**
of that slice. Replicas do two things: they share load, and they keep the model
servable when one holder disappears.

The capacity planner tracks how many replicas each slice has and reports the
weakest one. A target replication factor guides where new nodes are placed.

## States of a model

Given the current coverage, a model is in one of three states:

| State | Meaning |
|-------|---------|
| **Robust** | fully covered, every slice at or above the target replication |
| **Fragile** | fully covered, but some slice has only one holder |
| **Incomplete** | at least one layer range is held by nobody |

Only robust and fragile models are servable. An incomplete model is not, and a
request for one fails cleanly, naming the missing ranges.

## Placement

When a new node joins, the planner does not place it arbitrarily. In order of
priority it will:

1. **Fill a gap.** If a slice is missing entirely, cover it first.
2. **Reinforce the weakest slice.** If everything is covered but one slice has a
   single holder, add a second there.
3. **Add redundancy.** If coverage is already robust, strengthen the
   least-replicated slice.

## Failover during a request

Generation does not stop when a replica dies mid-request. Each pipeline stage
tries its replicas in turn: if the one it picked fails, it marks it dead and
falls through to the next holder of the same slice. As long as one replica of
each slice survives, the request completes.

## Next

- [NAT relay](/concepts/nat-relay)
- [Choosing a model](/guides/choosing-a-model)
