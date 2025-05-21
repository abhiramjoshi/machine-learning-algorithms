import random
import pprint
import numpy as np
import numpy.typing as npt
from ..utils import setup_logger
import logging

logger = setup_logger("kmeans")
logger.setLevel(logging.DEBUG)
logger.propagate = True

class K_Means():
    def __init__(self, vectors, k):
        self.vectors: list[npt.NDArray[np.float64]] = vectors
        self.N = len(self.vectors)
        self.k: int = k
        self.groups: list[list[int]] = [[] for _ in range(self.k)]
        self.representatives: list[npt.NDArray[np.float64]] = []
        self.j_group: list[float] = [np.inf]*k
        self.j: float = self.calculate_J_clust()
        self.j_interations: list[float] = []
# K-means clustering algorithm
# Find centroid
    def calculate_representatives(self) -> None:
        for i,group_indices in enumerate(self.groups):
            group = [self.vectors[j] for j in group_indices]
            z = self.calculate_z(group)
            self.representatives[i] = z

    def calculate_z(self, group: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        group_array = np.concatenate(group)
        logger.debug(f"Group Array: {group_array}")
        z = np.mean(group_array, axis=0)
        logger.debug(f"Z-vector: {z}")
        return z
        

# Allocate groups
    def allocate_group(self, x: npt.NDArray[np.float64]) -> tuple[int,float]:
        logger.info(f"Calculating group allocation for x: {x}")
        min_dist = np.inf
        min_index = -1

        for i,z in enumerate(self.representatives):
            dist = np.linalg.norm(x - z)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        
        logger.debug(f"X is part of group {min_index}| Distance: {min_dist}")
        return min_index, min_dist


# Get clustering coefficient
    def calculate_J_clust(self):
        logger.info("Calculating J_clust")
        J_total = 0
        for i,j in enumerate(self.j_group):
            J_group = j*len(self.groups[i])
            J_total += J_group
            logger.debug(f"J_group is: {J_group}. Group size: {len(self.groups[i])}")
        return J_total / self.N
# Calculate groups
    def calculate_groups(self) -> None:
        logger.info("Calculating group allocations")
        self.groups = [[] for _ in range(self.k)]
        self.j_group = [0.0]*self.k
        for i,x in enumerate(self.vectors):
            group, dist = self.allocate_group(x)
            self.j_group[group] += dist
            self.groups[group].append(i)

        logger.debug(f"Group Allocations:\n{self.groups}")
            

    def run_iteration(self):
        self.calculate_groups()
        self.calculate_representatives()
        iter_j = self.calculate_J_clust()
        self.j = iter_j
        return iter_j

    def run_clustering(self, thresh: float|None =0.1, n_iter: int|None =None) :
        # select random starting group representatives
        logger.info("Running k-means clustering")
        logger.info("Selecting random group representatives")
        self.representatives = random.sample(self.vectors, self.k)
        logger.debug(f"Group representatives chosen:\n{self.representatives}")
        if thresh:
            prev_J = np.inf
            j_diff = np.inf
            i = 0
            while j_diff >= thresh:
                J_iter = self.run_iteration()
                j_diff = abs(prev_J - J_iter)
                logger.debug(f"J_diff: {j_diff}")
                prev_J = J_iter
                self.j_interations.append(J_iter)
                logger.info(f"Iteration: {i} --> J_clust: {J_iter}")
                logger.info(self.__dict__)
                i += 1

            return self.j_interations[-1]

        if n_iter:
            for i in range(n_iter):
                J_iter = self.run_iteration()
                self.j_interations.append(J_iter)
                print(f"Iteration: {i} --> J_clust: {J_iter}")
                pprint.pprint(self.__dict__)

            return self.j_interations[-1]

        raise Exception("Either thresh or n_iter must be defined")

if __name__ == "__main__":
    coordinates = [np.array([[random.randint(1,20), random.randint(1,20)]]) for _ in range(20)]
    k = 4
    logger.info("Setting up k-means clustering demo")
    clustering = K_Means(vectors=coordinates, k=k)
    J = clustering.run_clustering()
    
    pprint.pprint(clustering.groups)
    print("J_clust: ", J)

