import numpy as np
import torch
from scipy.optimize import LinearConstraint, Bounds, milp

from flowmm.rfm.manifolds.flat_torus import FlatTorus01


def ot_reassignment(x0, x1, classes, cost="geodesic"):
    """Matches points in x0 to points in x1 using ot with strict one-to-one matching and classes constraint."""
    # Convert torch tensors to numpy arrays for compatibility with pot and scipy

    assert x0.shape[0] == x1.shape[0], (
        "x0 and x1 must have the same number of points!" ""
    )
    assert (
        x0.shape[-1] == 3
    ), "x0 and x1 must be georep, i.e. must have shape (n_points, 3)!"

    x0 = x0.cpu()
    x1 = x1.cpu()
    classes = classes.cpu().numpy()

    # Compute the cost matrix (pairwise squared Euclidean distance)
    if cost == "geodesic":
        cost_vectors = FlatTorus01.logmap(x0[:, None, :], x1[None, :, :])
        cost_matrix = (cost_vectors**2).sum(-1).sqrt()
        cost_matrix = cost_matrix.numpy()

    elif cost == "euclidean":
        x0 = x0.cpu().numpy()
        x1 = x1.cpu().numpy()
        cost_matrix = np.linalg.norm(x0[:, None, :] - x1[None, :, :], axis=-1)
    else:
        raise ValueError(f"Unknown cost function {cost}")

    c = cost_matrix.flatten()

    # indicate that the variables are integers {0, 1}
    integrality = np.ones(cost_matrix.shape[0] * cost_matrix.shape[1])
    x_lb = np.zeros_like(c)
    x_ub = np.ones_like(c)
    bounds = Bounds(x_lb, x_ub)

    # constraints
    A = []
    b_l = []
    b_u = []

    # torch.unique to make it more flexible
    # this way it doesnt matter if there are gaps in the classes
    for i in np.unique(classes):
        # assign the correct number of points from each classes
        mask = (classes == i).nonzero()[0]
        points_per_class = (classes == i).sum()
        class_constraint = np.zeros(cost_matrix.shape)
        class_constraint[mask[:, None], mask] = 1
        A.append(class_constraint.reshape((1, -1)))
        b_l.append(points_per_class)
        b_u.append(points_per_class)

        """
        Looks like this if i = 0
               b0 b0 b1 b1 b1  ...
               a1 a2 a3 a4 a5  ...
        b0 a1   1  1  0  0  0  ...   
        b0 a2   1  1  0  0  0  ...
        b1 a3   0  0  0  0  0  ...
        b1 a4   0  0  0  0  0  ...
        b1 a5   0  0  0  0  0
        ...

        if classes = 1
        Looks like this if i = 0
               b0 b0 b1 b1 b1  ...
               a1 a2 a3 a4 a5  ...
        b0 a1   0  0  0  0  0  ...   
        b0 a2   0  0  0  0  0  ...
        b1 a3   0  0  1  1  1  ...
        b1 a4   0  0  1  1  1  ...
        b1 a5   0  0  1  1  1
        ...

        e.g. b0 has 2 points, i.e. 2 assignments have to be made within the b0.
        this is enforced in b_l and b_u
        """

    for i, _ in enumerate(x0):
        # each point must be chosen only once
        point_constraint_i = np.zeros(cost_matrix.shape)
        point_constraint_i[i, :] = 1
        A.append(point_constraint_i.reshape((1, -1)))
        b_l.append(1)
        b_u.append(1)

        """
        Looks like this
            a0 a1 a2 a3  ...
        a1   1  1  1  1  ...   
        a2   0  0  0  0  ...
        a3   0  0  0  0  ...
        ...
        """

        point_constraint_j = np.zeros(cost_matrix.shape)
        point_constraint_j[:, i] = 1
        A.append(point_constraint_j.reshape((1, -1)))
        b_l.append(1)
        b_u.append(1)

        """
        Looks like this
            a0 a1 a2 a3  ...
        a1   1  0  0  0  ...   
        a2   1  0  0  0  ...
        a3   1  0  0  0  ...
        ...
        """

    A = np.vstack(A)
    b_l = np.array(b_l)
    b_u = np.array(b_u)

    constraints = LinearConstraint(A, b_l, b_u)

    # NOTE default is to minimize, so we're good with the cost matrix
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    best_assignment = res.x.reshape(cost_matrix.shape)
    _, col_indices = np.where(best_assignment == 1)

    cost = res.fun
    return cost, torch.tensor(col_indices, dtype=torch.long)


if __name__ == "__main__":
    # Test the OT matching
    x0 = (1 / 4) * torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    x1 = (1 / 4) * torch.tensor([[1, 1], [0, 0], [3, 3], [2, 2], [4, 4]])

    classes = torch.tensor([0, 0, 1, 1, 1])

    euclidean_result = ot_reassignment(x0, x1, classes, cost="euclidean")
    assert torch.all(euclidean_result == torch.tensor([1, 0, 3, 2, 4]))

    print(euclidean_result)

    x0 = torch.tensor([[0.95, 0.95], [0.9, 0.9], [0.85, 0.85], [0.1, 0.1], [0.2, 0.2]])
    x1 = torch.tensor([[0.025, 0.025], [0.1, 0.1], [0.3, 0.3], [0.8, 0.8], [0.9, 0.9]])
    classes = torch.tensor([0, 0, 0, 1, 1])

    print(ot_reassignment(x0, x1, classes, cost="geodesic"))
