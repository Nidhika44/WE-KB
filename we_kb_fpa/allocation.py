"""Shared target-only allocation and bounded, fully observed slot features."""
import numpy as np


def binary_greedy_allocation(tasks, users, edges, cloud, sinr, distances, net):
    """Latency-greedy local/nearest-edge decisions with the shared allocator."""
    n,m=len(tasks),len(edges)
    home=np.argmin(distances,axis=1)
    nominal_bw=1./max(1,n//m)
    targets=np.zeros(n,dtype=np.int32)
    for i,(task,user) in enumerate(zip(tasks,users)):
        edge=edges[home[i]]
        rate=max(1e3,nominal_bw*edge.bandwidth_total*np.log2(1.+sinr[i,home[i]]))
        remote=(task.data_size_bits/rate+task.total_cycles/min(edge.f_total,2e9)
                +edge.queue_delay_current+distances[i,home[i]]/3e8+net.d_dl_constant)
        if remote<task.total_cycles/user.f_local:
            targets[i]=home[i]+1
    return uniform_allocation(targets,tasks,users,edges,cloud,distances)


def uniform_allocation(targets, tasks, users, edges, cloud, distances):
    targets = np.asarray(targets, dtype=int).copy()
    n, m = len(tasks), len(edges)
    if targets.shape != (n,) or np.any((targets < 0) | (targets > m + 1)):
        raise ValueError("Invalid target vector")
    cpu, bw = np.zeros(n), np.zeros(n)
    local = targets == 0
    cpu[local] = np.array([u.f_local for u in users])[local]
    for target, capacity in enumerate([e.f_total for e in edges] + [cloud.f_total], 1):
        selected = targets == target
        if selected.any():
            cpu[selected] = capacity / selected.sum()
    access = targets - 1
    remote_cloud = targets == m + 1
    access[remote_cloud] = np.argmin(distances[remote_cloud], axis=1)
    for edge in range(m):
        selected = (~local) & (access == edge)
        if selected.any():
            bw[selected] = 1.0 / selected.sum()
    return targets, cpu, bw


def task_features(tasks, users, sinr, distances):
    return np.column_stack((
        np.log1p([task.data_size_bits / 1e6 for task in tasks]),
        np.log1p([task.total_cycles / 1e9 for task in tasks]),
        [task.deadline_sec for task in tasks], [u.f_local / 1e9 for u in users],
        [u.p_tx for u in users],
        np.clip(10 * np.log10(np.maximum(sinr, 1e-12)), -120, 120) / 60,
        np.log1p(distances / 100)))


def slot_features(tasks, users, edges, cloud, sinr, distances):
    rows = task_features(tasks, users, sinr, distances)
    context = np.array([e.f_total / 1e11 for e in edges] +
                       [e.bandwidth_total / 1e8 for e in edges] +
                       [e.queue_delay_current for e in edges] +
                       [cloud.f_total / 1e11, cloud.propagation_delay, cloud.queue_delay_current])
    features = np.concatenate((rows.ravel(), context))
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite slot observation")
    return features


def action_features(task, user, edges, cloud, sinr, distances, counts, radio_counts, net):
    """Observable latency/energy estimates before unassigned tasks contend."""
    cycles, bits = task.total_cycles, task.data_size_bits
    latencies = [cycles / user.f_local]
    energies = [net.kappa_local * user.f_local ** 2 * cycles]
    for index, edge in enumerate(edges):
        frequency = max(1e6, edge.f_total / (counts[index+1] + 1))
        rate = max(1e3, edge.bandwidth_total * np.log2(1 + sinr[index]) / (radio_counts[index] + 1))
        transmit = bits / rate
        latencies.append(transmit + cycles/frequency + edge.queue_delay_current + distances[index]/3e8 + net.d_dl_constant)
        energies.append(user.p_tx * transmit + net.kappa_edge * frequency ** 2 * cycles)
    index = int(np.argmin(distances))
    frequency = max(1e6, cloud.f_total / (counts[-1]+1))
    rate = max(1e3, edges[index].bandwidth_total * np.log2(1+sinr[index]) / (radio_counts[index]+1))
    transmit = bits/rate
    latencies.append(transmit + bits/cloud.backhaul_bandwidth + cycles/frequency + cloud.queue_delay_current + cloud.propagation_delay + net.d_dl_constant)
    energies.append(user.p_tx*transmit + net.kappa_cloud*frequency ** 2 * cycles)
    return np.concatenate((np.log1p(np.asarray(latencies)/task.deadline_sec),
                           np.log1p(np.asarray(energies)/max(energies[0], 1e-6))))
