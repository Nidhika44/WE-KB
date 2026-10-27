"""User-aligned full-slot memory for joint placement seeds; no future outcomes."""
from dataclasses import dataclass
import numpy as np
from we_kb_fpa.allocation import slot_features


@dataclass
class JointMemoryRecord:
    users: tuple
    edges: tuple
    features: np.ndarray
    targets: np.ndarray


class JointStateMemory:
    def __init__(self,capacity=1000):
        if capacity<1:
            raise ValueError('Memory capacity must be positive')
        self.capacity=capacity
        self.records=[]

    @staticmethod
    def observation(args):
        tasks,users,edges,cloud,sinr,distances=args
        ids=np.array([user.device_id for user in users])
        if len(set(ids))!=len(ids) or any(task.user_id!=user.device_id for task,user in zip(tasks,users)):
            raise ValueError('Task/user alignment must be explicit and unique')
        order=np.argsort(ids,kind='stable')
        features=np.r_[slot_features([tasks[i] for i in order],[users[i] for i in order],edges,cloud,sinr[order],distances[order]),cloud.backhaul_bandwidth/1e8]
        return tuple(int(ids[i]) for i in order),tuple(edge.server_id for edge in edges),order,features

    def add(self,args,solution):
        ids,edges,order,features=self.observation(args)
        self.records.append(JointMemoryRecord(ids,edges,features.copy(),solution.targets[order].copy()))
        if len(self.records)>self.capacity:
            self.records.pop(0)

    def query(self,args,k=3):
        ids,edges,order,features=self.observation(args)
        candidates=[]
        for index,record in enumerate(self.records):
            if record.users==ids and record.edges==edges and record.features.shape==features.shape:
                candidates.append((float(np.linalg.norm(record.features-features)),-index,record))
        result=[]
        for _,_,record in sorted(candidates,key=lambda item:(item[0],item[1]))[:k]:
            targets=np.empty(len(order),dtype=np.int32)
            targets[order]=record.targets
            result.append(targets)
        return result
