use std::collections::BTreeMap;

use crate::registry::PeerRegistry;

#[derive(Debug, Clone)]
pub struct SliceCoverage {
    pub start_layer: u32,
    pub end_layer: u32,
    pub replicas: usize,
}

#[derive(Debug, Clone)]
pub struct ModelCapacity {
    pub model_id: String,
    pub total_layers: u32,
    pub slices: Vec<SliceCoverage>,
    pub servable: bool,
    pub min_replicas: usize,
    pub weakest_slice: Option<(u32, u32)>,
}

impl ModelCapacity {
    pub fn is_robust(&self, target_replication: usize) -> bool {
        self.servable && self.min_replicas >= target_replication
    }

    pub fn missing_slices(&self) -> Vec<(u32, u32)> {
        self.slices
            .iter()
            .filter(|s| s.replicas == 0)
            .map(|s| (s.start_layer, s.end_layer))
            .collect()
    }

    /// The layer ranges of `[0, total_layers)` that no live slice covers.
    /// Unlike `missing_slices` (which only reports announced-but-unheld slices),
    /// this walks the whole model depth and reports every hole — including the
    /// tail region no peer advertises at all, e.g. `12:64` when the network only
    /// holds `0:12` of a 64-layer model. Empty when the model is fully servable.
    pub fn coverage_gaps(&self) -> Vec<(u32, u32)> {
        let mut gaps = Vec::new();
        if self.total_layers == 0 {
            return gaps;
        }
        let mut covered_up_to = 0u32;
        loop {
            // Extend the covered prefix as far as any live slice reaches.
            let mut progressed = true;
            while progressed {
                progressed = false;
                for s in &self.slices {
                    if s.replicas > 0 && s.start_layer <= covered_up_to && s.end_layer > covered_up_to
                    {
                        covered_up_to = s.end_layer;
                        progressed = true;
                    }
                }
            }
            if covered_up_to >= self.total_layers {
                break;
            }
            // There is a hole starting at `covered_up_to`. It runs until the next
            // live slice begins, or to the end of the model if none do.
            let next_start = self
                .slices
                .iter()
                .filter(|s| s.replicas > 0 && s.start_layer > covered_up_to)
                .map(|s| s.start_layer)
                .min()
                .unwrap_or(self.total_layers);
            gaps.push((covered_up_to, next_start));
            covered_up_to = next_start;
        }
        gaps
    }
}

pub fn analyze(registry: &PeerRegistry) -> Vec<ModelCapacity> {
    let peers = registry.all();

    let mut by_model: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for peer in peers {
        by_model.entry(peer.model_id.clone()).or_default().push(peer);
    }

    let mut result = Vec::new();

    for (model_id, model_peers) in by_model {
        // Prefer the model's true depth as reported by the worker (propagated
        // through gossip in `total_layers`). Fall back to inferring it from the
        // highest advertised `end_layer` only when no peer reports a real total
        // — i.e. every holder predates the field. Without this, a lone node
        // serving layers 0:6 of a 64-layer model would look complete (7/7)
        // instead of incomplete (7/64).
        let reported_total = model_peers
            .iter()
            .map(|p| p.total_layers)
            .filter(|&t| t > 0)
            .max();
        let total_layers = reported_total
            .unwrap_or_else(|| model_peers.iter().map(|p| p.end_layer).max().unwrap_or(0));

        let mut slice_counts: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for p in &model_peers {
            *slice_counts.entry((p.start_layer, p.end_layer)).or_insert(0) += 1;
        }

        let slices: Vec<SliceCoverage> = slice_counts
            .iter()
            .map(|(&(start, end), &count)| SliceCoverage {
                start_layer: start,
                end_layer: end,
                replicas: count,
            })
            .collect();

        let servable = is_fully_covered(&slices, total_layers);

        let min_replicas = if servable {
            slices.iter().map(|s| s.replicas).min().unwrap_or(0)
        } else {
            0
        };

        let weakest_slice = slices
            .iter()
            .min_by_key(|s| s.replicas)
            .map(|s| (s.start_layer, s.end_layer));

        result.push(ModelCapacity {
            model_id,
            total_layers,
            slices,
            servable,
            min_replicas,
            weakest_slice,
        });
    }

    result
}

fn is_fully_covered(slices: &[SliceCoverage], total_layers: u32) -> bool {
    if total_layers == 0 {
        return false;
    }
    let mut covered_up_to = 0u32;
    let mut progressed = true;
    while covered_up_to < total_layers && progressed {
        progressed = false;
        for s in slices {
            if s.replicas > 0 && s.start_layer <= covered_up_to && s.end_layer > covered_up_to {
                covered_up_to = s.end_layer;
                progressed = true;
            }
        }
    }
    covered_up_to >= total_layers
}

#[derive(Debug, Clone, PartialEq)]
pub enum Placement {
    FillMissing { start: u32, end: u32 },
    ReinforceWeak { start: u32, end: u32, current_replicas: usize },
    ModelNotPresent,
}

fn first_coverage_gap(cap: &ModelCapacity) -> Option<(u32, u32)> {
    let mut covered_up_to = 0u32;
    let mut progressed = true;
    while covered_up_to < cap.total_layers && progressed {
        progressed = false;
        for s in &cap.slices {
            if s.replicas > 0 && s.start_layer <= covered_up_to && s.end_layer > covered_up_to {
                covered_up_to = s.end_layer;
                progressed = true;
            }
        }
    }
    if covered_up_to < cap.total_layers {
        let next_start = cap
            .slices
            .iter()
            .filter(|s| s.replicas > 0 && s.start_layer > covered_up_to)
            .map(|s| s.start_layer)
            .min()
            .unwrap_or(cap.total_layers);
        Some((covered_up_to, next_start))
    } else {
        None
    }
}

pub fn recommend_placement(
    caps: &[ModelCapacity],
    model_id: &str,
    target_replication: usize,
) -> Placement {
    let cap = match caps.iter().find(|c| c.model_id == model_id) {
        Some(c) => c,
        None => return Placement::ModelNotPresent,
    };

    if let Some((start, end)) = first_coverage_gap(cap) {
        return Placement::FillMissing { start, end };
    }

    let weakest = cap
        .slices
        .iter()
        .filter(|s| s.replicas < target_replication)
        .min_by_key(|s| s.replicas);

    if let Some(s) = weakest {
        return Placement::ReinforceWeak {
            start: s.start_layer,
            end: s.end_layer,
            current_replicas: s.replicas,
        };
    }

    let least = cap.slices.iter().min_by_key(|s| s.replicas).unwrap();
    Placement::ReinforceWeak {
        start: least.start_layer,
        end: least.end_layer,
        current_replicas: least.replicas,
    }
}

pub fn best_servable_model(caps: &[ModelCapacity]) -> Option<&ModelCapacity> {
    caps.iter()
        .filter(|c| c.servable)
        .max_by_key(|c| c.total_layers)
}

pub fn fallback_after_loss<'a>(
    caps: &'a [ModelCapacity],
    current_model: &str,
) -> Option<&'a ModelCapacity> {
    let current_servable = caps
        .iter()
        .find(|c| c.model_id == current_model)
        .map(|c| c.servable)
        .unwrap_or(false);

    if current_servable {
        return None;
    }
    best_servable_model(caps)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceAssignment {
    pub start: u32,
    pub end: u32,
    pub reason: String,
}

pub fn assign_slice(
    caps: &[ModelCapacity],
    model_id: &str,
    total_layers: u32,
    my_capacity_layers: u32,
    target_replication: usize,
) -> Option<SliceAssignment> {
    if my_capacity_layers == 0 || total_layers == 0 {
        return None;
    }
    let cap_layers = my_capacity_layers.min(total_layers);

    let existing = caps.iter().find(|c| c.model_id == model_id);

    // Case 1: model not present at all -> take from the start.
    let Some(cap) = existing else {
        let end = cap_layers.min(total_layers);
        return Some(SliceAssignment {
            start: 0,
            end,
            reason: "bootstrap: first holder of this model".to_string(),
        });
    };

    // Case 2: there is a coverage gap -> fill it (as much as capacity allows).
    if let Some((gap_start, gap_end)) = first_coverage_gap_with_total(cap, total_layers) {
        let want = gap_end - gap_start;
        let take = want.min(cap_layers);
        return Some(SliceAssignment {
            start: gap_start,
            end: gap_start + take,
            reason: format!("fill coverage gap {}:{}", gap_start, gap_end),
        });
    }

    // Case 3: fully covered -> reinforce the weakest slice if under target.
    if let Some(weak) = cap
        .slices
        .iter()
        .filter(|s| s.replicas < target_replication)
        .min_by_key(|s| s.replicas)
    {
        let want = weak.end_layer - weak.start_layer;
        if want <= cap_layers {
            return Some(SliceAssignment {
                start: weak.start_layer,
                end: weak.end_layer,
                reason: format!(
                    "reinforce fragile slice {}:{} ({} replica)",
                    weak.start_layer, weak.end_layer, weak.replicas
                ),
            });
        }
    }

    // Case 4: everything robust -> add redundancy on the least-replicated slice.
    let least = cap.slices.iter().min_by_key(|s| s.replicas)?;
    let want = least.end_layer - least.start_layer;
    if want <= cap_layers {
        Some(SliceAssignment {
            start: least.start_layer,
            end: least.end_layer,
            reason: "add redundancy".to_string(),
        })
    } else {
        None
    }
}

fn first_coverage_gap_with_total(cap: &ModelCapacity, total_layers: u32) -> Option<(u32, u32)> {
    let mut covered_up_to = 0u32;
    let mut progressed = true;
    while covered_up_to < total_layers && progressed {
        progressed = false;
        for s in &cap.slices {
            if s.replicas > 0 && s.start_layer <= covered_up_to && s.end_layer > covered_up_to {
                covered_up_to = s.end_layer;
                progressed = true;
            }
        }
    }
    if covered_up_to < total_layers {
        let next_start = cap
            .slices
            .iter()
            .filter(|s| s.replicas > 0 && s.start_layer > covered_up_to)
            .map(|s| s.start_layer)
            .min()
            .unwrap_or(total_layers);
        Some((covered_up_to, next_start))
    } else {
        None
    }
}