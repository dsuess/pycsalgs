# encoding: utf-8

import numpy as np

import cgen as c
import cgen.cuda as ccu


DGEMV_SRC = c.LiteralLines("""
typedef enum {blasTranspose, blasNoTranspose} BlasTranspose;


/* Compute the matrix-vector product C = A @ B

Parameters
----------

trans: determines whether we compute A @ B or A.T @ B. The following documentation
       assumes trans == blasNoTranspose (size of matrices may vary)

A:     (M, N) matrix of type double in row-major format

B:     (N,) vector of type double

C:     (M,) vector of type double, result will be written to this

TODO
----
- Replace this by an actual cuBlas call.

*/
__device__ static void dgemv(BlasTranspose const trans,
                             double const *const A,
                             double const *const B,
                             int const M,
                             int const N,
                             double *const C)
{
    int m, n;
    switch (trans) {
        case blasNoTranspose:
            for (m = 0; m < M; ++m) {
                double prod = 0.0;
                for (n = 0; n < N; ++n) {
                    prod += A[m * N + n] * B[n];
                }
                C[m] = prod;
            }
            break;

        case blasTranspose:
            for (n = 0; n < N; ++n) {
                double prod = 0.0;
                for (m = 0; m < M; ++m) {
                    prod += A[m * N + n] * B[m];
                }
                C[n] = prod;
            }
            break;
    }

}

""")


def ConstPointerToConst(dtype, name, value):
    """Returns a cgen variable declaration & assignment of a constant pointer to
    a constant of type `dtype`
    """
    return c.Constant(c.Pointer(c.Const(c.POD(dtype, name))), value)


def ConstPointerToConstDecl(dtype, name):
    """Returns a cgen variable declaration of a constant pointer to a constant
    of type `dtype`
    """
    return c.Const(c.Pointer(c.Const(c.POD(dtype, name))))


class AMECodeGenCuda(object):
    """A CUDA code generator for the alternating minimization algorithm.

    """

    def __init__(self, dims, ranks, meas, cuda_blocksize=512, dtype=np.float_):
        """@todo: to be defined1. """
        assert len(dims) == len(ranks) + 1
        assert all(np.isscalar(d) for d in dims)

        self._sites = len(dims)
        self._dims = dims
        self._ranks = ranks
        self._ltens_sizes = [(1 if i == 0 else ranks[i - 1])
                             * (1 if i == self._sites - 1 else self._ranks[i])
                             * dims[i] for i in range(self._sites)]
        self._meas = meas
        self._cuda_bs = cuda_blocksize
        self._dtype = dtype

    def copy_ltens_to_share(self, pos):
        """Generates the instructions which copy the `pos`-th local tensor of
        `X` to the shared variable `x_shared` (which should be initialized at
        this point)

        :param pos: The local tensor to copy (should be `< len(X)`)
        :returns: List containing cgen Statements

        """
        elems_to_copy = self._ltens_sizes[pos]
        copy_src = 'x_shared[{bs:d} * {{batch:d}} + threadIdx.x] = X{pos:d}[{bs:d} * {{batch:d}} + threadIdx.x]' \
            .format(bs=self._cuda_bs, pos=pos)

        statements = [c.Statement('__syncthreads()')]
        statements += [c.Statement(copy_src.format(batch=batch))
                    for batch in range(elems_to_copy // self._cuda_bs)]

        if elems_to_copy % self._cuda_bs != 0:
            start_elem = (elems_to_copy // self._cuda_bs) * self._cuda_bs
            rest_elems = elems_to_copy - start_elem
            batch = elems_to_copy // self._cuda_bs
            statement = c.If('threadIdx.x < %i' % rest_elems,
                            c.Statement(copy_src.format(batch=batch)))
            statements += [statement]

        statements += [c.Statement('__syncthreads()')]
        return statements

    def declaration(self, pos):
        """Generates the declarative instructions for the optimizations over
        sites nr. `pos`

        :param pos: The local tensor to copy (should be `< len(X)`)
        :returns: List containing cgen Statements
        """
        max_ltens_size = max(self._ltens_sizes)
        max_left_size = 1 if pos == 0 else max(self._ranks[:pos])
        max_right_size = 1 if pos == self._sites - 1 else max(self._ranks[pos:])
        max_tmat_size = max(self._ranks[i] * self._ranks[i + 1]
                            for i in range(self._sites - 2))

        init_statements = [
            c.LineComment("Define the row number the current thread is operating on"),
            c.Initializer(c.Const(c.POD(np.int32, 'mid')),
                          'threadIdx.x + blockIdx.x * blockDim.x'),

            c.LineComment("Allocate shared memory for the local tensors"),
            ccu.CudaShared(c.ArrayOf(c.POD(self._dtype, 'x_shared'), max_ltens_size)),

            c.LineComment("Allocate the left-, right-, and transfer contractions"),
            c.ArrayOf(c.POD(self._dtype, 'left_c'), max_left_size),
            c.ArrayOf(c.POD(self._dtype, 'right_c'), max_right_size),
            c.ArrayOf(c.POD(self._dtype, 'tmat_c'), max_tmat_size),
            c.ArrayOf(c.POD(self._dtype, 'buf_c'), max(max_right_size, max_left_size)),

            c.LineComment("Shortcut for current row of design matrix"),
            c.LineComment("Carefull, current_row might be out of bounds!"),
            ConstPointerToConst(self._dtype, 'current_row', 'A + (mid * %i)'
                                % sum(self._ranks))
        ]

        return init_statements

    def left_contractions(self, pos):
        """Generates the code computing the left-contraction part of the
        opimization matrix for site nr. `pos`

        :param pos: The local tensor to copy (should be `< len(X)`)
        :returns: List containing cgen Statements

        """
        if pos == 0:
            return [c.Statement('left_c[0] = 1')]

        result = self.copy_ltens_to_share(0)
        result += [c.Line()]

        contract_ltens_with_a = 'dgemv(blasNoTranspose, x_shared, current_row + {offset:d}, {dim_out:d}, {dim_in:d}, {target:})'
        src = contract_ltens_with_a.format(offset=0, dim_out=self._ranks[0],
                                           dim_in=self._dims[0], target='left_c')
        # We need to check this every time and can't simpy return since
        # otherwise __syncthreads crashes
        result += [c.If('mid < %i' % self._meas, c.Statement(src))]

        for i in range(1, pos):
            result += self.copy_ltens_to_share(i)
            result += [c.Line()]

            # Since we assume A to consist of product measurements
            result += [
                c.If('mid < %i' % self._meas, c.Block([
                    c.Statement(contract_ltens_with_a
                                .format(offset=sum(self._dims[:i]),
                                        dim_out=self._ranks[i - 1] * self._ranks[i],
                                        dim_in=self._dims[i], target='tmat_c')),
                    c.Statement('dgemv(blasTranspose, tmat_c, left_c, {rank_l}, {rank_r}, buf_c)'
                                .format(rank_l=self._ranks[i - 1],
                                        rank_r=self._ranks[i])),
                    c.Statement('memcpy(left_c, buf_c, sizeof({ctype}) * {rank_r})'
                                .format(ctype=c.dtype_to_ctype(self._dtype),
                                        rank_r=self._ranks[i]))
                ])),
                c.Line()
            ]
        return result

    def right_contractions(self, pos):
        """Generates the code computing the right-contraction part of the
        opimization matrix for site nr. `pos`

        :param pos: The local tensor to copy (should be `< len(X)`)
        :returns: List containing cgen Statements

        """
        if pos == self._sites - 1:
            return [c.Statement('right_c[0] = 1')]

        result = self.copy_ltens_to_share(self._sites - 1)
        result += [c.Line()]

        contract_ltens_with_a = 'dgemv(blasNoTranspose, x_shared, current_row + {offset:d}, {dim_out:d}, {dim_in:d}, {target:})'
        src = contract_ltens_with_a.format(offset=sum(self._dims[:-1]),
                                           dim_out=self._ranks[-1],
                                           dim_in=self._dims[-1],
                                           target='right_c')
        result += [c.If('mid < %i' % self._meas, c.Statement(src))]

        for i in range(self._sites - 2, pos, -1):
            result += self.copy_ltens_to_share(i)
            result += [c.Line()]

            # Since we assume A to consist of product measurements
            result += [
                c.If('mid < %i' % self._meas, c.Block([
                    c.Statement(contract_ltens_with_a
                                .format(offset=sum(self._dims[:i]),
                                        dim_out=self._ranks[i - 1] * self._ranks[i],
                                        dim_in=self._dims[i], target='tmat_c')),
                    c.Statement('dgemv(blasNoTranspose, tmat_c, right_c, {rank_l}, {rank_r}, buf_c)'
                                .format(rank_l=self._ranks[i - 1],
                                        rank_r=self._ranks[i])),
                    c.Statement('memcpy(right_c, buf_c, sizeof({ctype}) * {rank_l})'
                                .format(ctype=c.dtype_to_ctype(self._dtype),
                                        rank_l=self._ranks[i - 1])),
                ])),
                c.Line()
            ]

        return result

    def generate_optimmat_code(self, pos, name=None):
        """Generates the code for computing the local optimization matrix
        for the optimization over site nr. `pos`

        The function has the following signature:

            DTYPE const *const A,
            DTYPE const *const X_0,
            ...,
            DTYPE const *const X_N,
            DTYPE *const result

        :param pos: The local tensor to copy (should be `< len(X)`)
        :param name: Name of the C function (default: get_optimmat_%(pos))
        :returns: cgen.FunctionBody with given name

        """
        name = 'get_optimmat_%i' % pos if name is None else name

        finalization_src = '''
        if (mid < {nr_meas:d}) {{
            for (uint i = 0; i < {pdim:d}; ++i) {{
                for (uint k_l = 0; k_l < {rank_l:d}; ++k_l) {{
                    for (uint k_r = 0; k_r < {rank_r:d}; ++k_r) {{
                        result[mid * {rank_l:d} * {pdim:d} * {rank_r:d}
                            + k_l * {pdim:d} * {rank_r:d}
                            + i * {rank_r:d}
                            + k_r]
                        = left_c[k_l] * current_row[{offset:d} + i] * right_c[k_r];
                    }}
                }}
            }}
        }}
        '''.format(nr_meas=self._meas, pdim=self._dims[pos],
                   rank_l=1 if pos == 0 else self._ranks[pos - 1],
                   rank_r=1 if pos == self._sites - 1 else self._ranks[pos],
                   offset=sum(self._dims[:pos]))
        finalization = c.LiteralLines(finalization_src)

        arg_decls = [ConstPointerToConstDecl(self._dtype, 'A')]
        arg_decls += [ConstPointerToConstDecl(self._dtype, 'X%i' % i)
                      for i in range(self._sites)]
        arg_decls += [c.Pointer(c.Const(c.POD(self._dtype, 'result')))]

        return c.FunctionBody(
            ccu.CudaGlobal(c.FunctionDeclaration(c.Value('void', 'get_optimmat_%i' % pos),
                                                 arg_decls=arg_decls)),
            c.Block(self.declaration(pos) + self.left_contractions(pos) +
                    self.right_contractions(pos) + [finalization])
        )

    def generate(self):
        """@todo: Docstring for generate.
        :returns: @todo

        """
        return c.Module([DGEMV_SRC] +
                        [self.generate_optimmat_code(pos) for pos in range(self._sites)])
