import torch
import torch.nn as nn


# Block Network

class SimpleBlockTensorNetwork(nn.Module):
    def __init__(self, outer_dim_s, outer_dim_r, outer_dim_o, inner_dim_s, inner_dim_r, inner_dim_o, with_nn = False):
        super(SimpleBlockTensorNetwork, self).__init__()

        self.outer_dim_s = outer_dim_s
        self.outer_dim_r = outer_dim_r
        self.outer_dim_o = outer_dim_o

        self.inner_dim_s = inner_dim_s
        self.inner_dim_r = inner_dim_r
        self.inner_dim_o = inner_dim_o

        self.with_nn = with_nn

        if with_nn:
            self.init_nn()
            self.first_param = inner_dim_r
        else:
            self.first_param = outer_dim_r

        self.init_tensors()


    def _core(self):
        return self.core_tensor


    def init_nn(self):
        self.lin21 = nn.Linear(self.outer_dim_r, self.inner_dim_r)
        nn.init.xavier_uniform_(self.lin21.weight)
        nn.init.zeros_(self.lin21.bias)

        self.relu1 = nn.ReLU()

        self.lin22 = nn.Linear(self.inner_dim_r, self.inner_dim_r)
        nn.init.xavier_uniform_(self.lin22.weight)
        nn.init.zeros_(self.lin22.bias)

        self.relu2 = nn.ReLU()

        self.lin23 = nn.Linear(self.inner_dim_r, self.inner_dim_r)
        nn.init.xavier_uniform_(self.lin23.weight)
        nn.init.zeros_(self.lin23.bias)


    def init_tensors(self):

        self.core_tensor = nn.Parameter(torch.randn(self.inner_dim_s, self.inner_dim_r, self.inner_dim_o)) # sro

        self.proj_s = nn.Parameter(torch.randn(self.outer_dim_s, self.inner_dim_s)) # Ss
        self.proj_r = nn.Parameter(torch.randn(self.first_param, self.inner_dim_r)) # Rr
        self.proj_o = nn.Parameter(torch.randn(self.outer_dim_o, self.inner_dim_o)) # Oo

        nn.init.xavier_uniform_(self.core_tensor)

        nn.init.xavier_uniform_(self.proj_s)
        nn.init.xavier_uniform_(self.proj_r)
        nn.init.xavier_uniform_(self.proj_o)


    def three_vectors_to_scalar(self, input1, input2, input3):
        if self.with_nn:
            input2 = self.lin21(input2)
            input2 = self.relu1(input2)
            input2 = self.lin22(input2)
            input2 = self.relu2(input2)
            input2 = self.lin23(input2)
        # Step 1: Project inputs to k dimensions
        p1 = torch.einsum('Ss,bS->bs', self.proj_s, input1) # bs
        p2 = torch.einsum('Rr,bR->br', self.proj_r, input2) # br
        p3 = torch.einsum('Oo,bO->bo', self.proj_o, input3) # bo

        # Step 2: Contract with the core tensor
        # Batch-wise contraction for a scalar output
        output = torch.einsum('bs,br,bo,sro->b', p1, p2, p3, self._core())
        return output


    def subject_to_matrix(self, input1):
        p1 = torch.einsum('Ss,bS->bs', self.proj_s, input1) # bs
        core_matrix = torch.einsum('sro,bs->bro', self._core(), p1) # bro
        matrix = torch.einsum('Rr,bro,Oo->bRO', self.proj_r, core_matrix, self.proj_o)
        assert matrix.shape == (input1.size(0), self.outer_dim_r, self.outer_dim_o)
        return matrix


    def relation_to_matrix(self, input2):
        if self.with_nn:
            input2 = self.lin21(input2)
            input2 = self.relu1(input2)
            input2 = self.lin22(input2)
            input2 = self.relu2(input2)
            input2 = self.lin23(input2)
        # Step 1: Project inputs to k dimensions
        p2 = torch.einsum('Rr,bR->br', self.proj_r, input2) # br
        core_matrix = torch.einsum('ors,br->bso', self._core(), p2) # bso
        matrix = torch.einsum('Ss,bos,Oo->bOS', self.proj_s, core_matrix, self.proj_o)

        batch_size = input2.size(0)
        assert matrix.shape == (batch_size, self.outer_dim_o, self.outer_dim_s)

        return matrix


    def object_to_matrix(self, input3):
        p3 = torch.einsum('Oo,bO->bo', self.proj_o, input3) # bo
        core_matrix = torch.einsum('sro,bo->bsr', self._core(), p3) # bsr
        matrix = torch.einsum('Ss,bsr,Rr->bSR', self.proj_s, core_matrix, self.proj_r)

        batch_size = input3.size(0)
        assert matrix.shape == (batch_size, self.outer_dim_s, self.outer_dim_r)

        return matrix



class SimpleBlockNetworkSubjectToMatrix(SimpleBlockTensorNetwork):
    forward = SimpleBlockTensorNetwork.subject_to_matrix

class SimpleBlockNetworkRelationToMatrix(SimpleBlockTensorNetwork):
    forward = SimpleBlockTensorNetwork.relation_to_matrix

class SimpleBlockNetworkObjectToMatrix(SimpleBlockTensorNetwork):
    forward = SimpleBlockTensorNetwork.object_to_matrix

class SimpleBlockNetworkVectorThreeVectorsToScalar(SimpleBlockTensorNetwork):
    forward = SimpleBlockTensorNetwork.three_vectors_to_scalar



class IndividualMatricesBlockNetworkRelationToMatrix(SimpleBlockTensorNetwork):
    forward = SimpleBlockTensorNetwork.relation_to_matrix

    def __init__(self, outer_dim_s, outer_dim_r, outer_dim_o, with_nn = False, rank=None):
        inner_dim_r = outer_dim_r
        inner_dim_s = outer_dim_s
        inner_dim_o = outer_dim_o
        self.rank = rank

        super(IndividualMatricesBlockNetworkRelationToMatrix, self).__init__(outer_dim_s, outer_dim_r, outer_dim_o, inner_dim_s, inner_dim_r, inner_dim_o, with_nn = with_nn)

        if self.rank is None:
            nn.init.xavier_uniform_(self.core_tensor)


    def init_tensors(self):
        device= "cuda"

        if self.rank is not None:
            self.U = nn.Parameter(torch.randn(self.inner_dim_r, self.inner_dim_s, self.rank))
            self.V = nn.Parameter(torch.randn(self.inner_dim_r, self.inner_dim_o, self.rank))

            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        else:
            self.core_tensor = nn.Parameter(torch.randn(self.inner_dim_s, self.inner_dim_r, self.inner_dim_o))

        self.proj_s = torch.eye(self.outer_dim_s, self.inner_dim_s).to(device)
        self.proj_r = torch.eye(self.first_param, self.inner_dim_r).to(device)
        self.proj_o = torch.eye(self.outer_dim_o, self.inner_dim_o).to(device)


    def _core(self):
        if self.rank is not None:
            # U  r s k   ,  V  r o k   ->  s r o   (then permute r to middle)
            core_rso = torch.einsum('rsk,rok->rso', self.U, self.V)
            return core_rso.permute(1, 0, 2).contiguous()          # s r o
        else:
            return self.core_tensor



# Triangle Network

class TriangleTensorNetwork(nn.Module):
    def __init__(self, outer_dim, inn_s, inn_r, inn_o, inn_x,inn_y,inn_z, with_nn):
        super(TriangleTensorNetwork, self).__init__()
        self.outer_dim = outer_dim
        self.inn_s = inn_s #s
        self.inn_r = inn_r #r
        self.inn_o = inn_o #o
        self.inn_x = inn_x #x
        self.inn_y = inn_y #y
        self.inn_z = inn_z #z

        self.with_nn = with_nn

        if with_nn:
            self.init_nn()
            self.first_param = inn_r
        else:
            self.first_param = outer_dim

        self.init_tensors()


    def init_tensors(self):
        self.cube_s = nn.Parameter(torch.randn(self.inn_s, self.inn_y, self.inn_z)) # syz
        self.cube_r = nn.Parameter(torch.randn(self.inn_x, self.inn_r, self.inn_z)) # xrz
        self.cube_o = nn.Parameter(torch.randn(self.inn_x, self.inn_y, self.inn_o)) # xyo

        self.proj_s = nn.Parameter(torch.randn(self.outer_dim, self.inn_s)) # Ss
        self.proj_r = nn.Parameter(torch.randn(self.first_param, self.inn_r)) # Rr
        self.proj_o = nn.Parameter(torch.randn(self.outer_dim, self.inn_o)) # Oo

        nn.init.xavier_uniform_(self.cube_s)
        nn.init.xavier_uniform_(self.cube_r)
        nn.init.xavier_uniform_(self.cube_o)

        nn.init.xavier_uniform_(self.proj_s)
        nn.init.xavier_uniform_(self.proj_r)
        nn.init.xavier_uniform_(self.proj_o)


    def init_nn(self):
        self.lin21 = nn.Linear(self.outer_dim, self.inn_r)
        nn.init.xavier_uniform_(self.lin21.weight)
        nn.init.zeros_(self.lin21.bias)
        self.relu1 = nn.ReLU()
        self.lin22 = nn.Linear(self.inn_r, self.inn_r)
        nn.init.xavier_uniform_(self.lin22.weight)
        nn.init.zeros_(self.lin22.bias)
        self.relu2 = nn.ReLU()
        self.lin23 = nn.Linear(self.inn_r, self.inn_r)
        nn.init.xavier_uniform_(self.lin23.weight)
        nn.init.zeros_(self.lin23.bias)


    def three_vectors_to_scalar(self, input1, input2, input3):
        if self.with_nn:
            input2 = self.lin21(input2)
            input2 = self.relu1(input2)
            input2 = self.lin22(input2)
            input2 = self.relu2(input2)
            input2 = self.lin23(input2)
        # Step 1: Project inputs to k dimensions
        p1 = torch.einsum('Ss,bS->bs', self.proj_s, input1) # bs
        p2 = torch.einsum('Rr,bR->br', self.proj_r, input2) # br
        p3 = torch.einsum('Oo,bO->bo', self.proj_o, input3) # bo


        output = torch.einsum('bs,br,bo,syz,xrz,xyo->b', p1, p2, p3, self.cube_s,self.cube_r, self.cube_o)

        return output


    def subject_to_matrix(self, input1):
        p1 = torch.einsum('Ss,bS->bs', self.proj_s, input1)
        core_matrix = torch.einsum('syz,xrz,xyo,bs->bro', self.cube_s, self.cube_r, self.cube_o, p1)  # bro
        matrix = torch.einsum('Rr,bro,Oo->bOR', self.proj_r, core_matrix, self.proj_o)
        batch_size, outer_dim = input1.size(0), self.outer_dim
        assert matrix.shape == (batch_size, outer_dim, outer_dim)
        return matrix


    def relation_to_matrix(self, input2):
        if self.with_nn:
            input2 = self.lin21(input2)
            input2 = self.relu1(input2)
            input2 = self.lin22(input2)
            input2 = self.relu2(input2)
            input2 = self.lin23(input2)

        p2 = torch.einsum('Rr,bR->br', self.proj_r, input2)
        core_matrix = torch.einsum('syz,xrz,xyo,br->bso', self.cube_s, self.cube_r, self.cube_o, p2)  # bso
        matrix = torch.einsum('Ss,bso,Oo->bOS', self.proj_s, core_matrix, self.proj_o)
        batch_size, outer_dim = input2.size(0), self.outer_dim
        assert matrix.shape == (batch_size, outer_dim, outer_dim)
        return matrix


    def object_to_matrix(self, input3):
        p3 = torch.einsum('Oo,bO->bo', self.proj_o, input3)
        core_matrix = torch.einsum('syz,xrz,xyo,bo->bsr', self.cube_s, self.cube_r, self.cube_o, p3)  # bsr
        matrix = torch.einsum('Ss,bsr,Rr->bRS', self.proj_s, core_matrix, self.proj_r)
        batch_size, outer_dim = input3.size(0), self.outer_dim
        assert matrix.shape == (batch_size, outer_dim, outer_dim)
        return matrix



class TriangleNetworkSubjectToMatrix(TriangleTensorNetwork):
    forward = TriangleTensorNetwork.subject_to_matrix

class TriangleNetworkRelationToMatrix(TriangleTensorNetwork):
    forward = TriangleTensorNetwork.relation_to_matrix

class TriangleNetworkObjectToMatrix(TriangleTensorNetwork):
    forward = TriangleTensorNetwork.object_to_matrix

class TriangleNetworkThreeVectorsToScalar(TriangleTensorNetwork):
    forward = TriangleTensorNetwork.three_vectors_to_scalar


if __name__ == "__main__":

    block = TriangleTensorNetwork(outer_dim = 200, inn_s = 10, inn_r = 20, inn_o = 30, inn_x = 1,inn_y = 2,inn_z = 3, with_nn = False)
    input1 = torch.reshape(torch.randn(200), (1, 200))
    input2 = torch.reshape(torch.randn(200), (1, 200))
    input3 = torch.reshape(torch.randn(200), (1, 200))
    block.subject_to_matrix(input1)
    block.relation_to_matrix(input2)
    block.object_to_matrix(input3)
