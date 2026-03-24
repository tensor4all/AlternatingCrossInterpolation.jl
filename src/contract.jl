function contract(matrixop, A, Aindices, B, Bindices; indexpermutation=[])
    if length(Aindices) != length(Bindices)
        error("Different number of legs to contract for tensors A and B.")
    end
    if any(Aindices .< 1) || any(Aindices .> ndims(A))
        error("Got indices out of range for tensor A in contract().")
    end
    if any(Bindices .< 1) || any(Bindices .> ndims(B))
        error("Got indices out of range for tensor B in contract().")
    end

    # indices of legs *not* to be contracted
    Arestinds = setdiff(1:ndims(A), Aindices)
    Brestinds = setdiff(1:ndims(B), Bindices)

    # reshape tensors into matrices with "thick" legs
    A2 = reshape(
        permutedims(A, tuple(cat(dims=1, Arestinds, Aindices))...),
        (prod(size(A)[Arestinds]), prod(size(A)[Aindices]))
    )
    B2 = reshape(
        permutedims(B, tuple(cat(dims=1, Bindices, Brestinds))...),
        (prod(size(B)[Bindices]), prod(size(B)[Brestinds]))
    )
    C2 = matrixop(A2, B2) # matrix multiplication

    # size of C
    Cdim = (size(A)[Arestinds]..., size(B)[Brestinds]...)
    # reshape matrix to tensor
    C = reshape(C2, Cdim)

    if !isempty(indexpermutation) # if permutation option is given
        C = permutedims(C, indexpermutation)
    end

    return C
end

"""
    contract(A, Aindices, B, Bindices, indexpermutation=[])

Contract tensors `A` and `B`.
The legs to be contracted are given by `Aindices` and `Bindices`.

Input:
- `A`, `B` (numeric arrays): the tensors to be contracted.
- `Aindices`, `Bindices` (ordered collection of integers):
    Indices for the legs of `A` and `B` to be contracted. The `Aindices[n]`-th leg of `A`
    will be contracted with the `Bindices[n]`-th leg of `B`. `Aindices` and `Bindices`
    should have the same size. If they are both empty, the result will be the direct product
    of `A` and `B`.
- `indexpermutation` (ordered collection of integers):
    Permutation of of the output tensor's indices to be performed after contraction.

Output:
- Contraction of `A` and `B`. All non-contracted legs are in the following order: first, all
    legs previously attached to `A`, in order, then all legs previously attached to `B`, in
    order.
"""
function contract(A, Aindices, B, Bindices; indexpermutation=[])
    return contract(*, A, Aindices, B, Bindices; indexpermutation)
end

_reversedims(T, i) = (ndims(T) + 1) .- i

function contractonchain(A, Aindices, B, Bindices; swap::Bool)
    if swap
        contract(B, _reversedims(B, Bindices), A, _reversedims(A, Aindices))
    else
        contract(A, Aindices, B, Bindices)
    end
end
