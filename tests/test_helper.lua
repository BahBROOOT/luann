local T = {}

function T.close(a, b, eps, label)
    eps = eps or 1e-6
    if math.abs(a - b) > eps then
        error(("%s: expected %.9f ~= %.9f"):format(label or "value", a, b), 2)
    end
end

function T.matrix_close(A, B, eps, label)
    label = label or "matrix"
    assert(type(A) == "table", label .. ": left side is not a table")
    assert(type(B) == "table", label .. ": right side is not a table")
    assert(#A == #B, label .. ": row count mismatch")
    assert(#A == 0 or #A[1] == #B[1], label .. ": col count mismatch")
    for i=1,#A do
        for j=1,#A[i] do
            T.close(A[i][j], B[i][j], eps, ("%s[%d,%d]"):format(label, i, j))
        end
    end
end

function T.vector_equal(A, B, label)
    label = label or "vector"
    assert(#A == #B, label .. ": length mismatch")
    for i=1,#A do
        assert(A[i] == B[i], ("%s[%d]: expected %s, got %s"):format(label, i, tostring(B[i]), tostring(A[i])))
    end
end

function T.prob_rows(A, eps, label)
    label = label or "probabilities"
    eps = eps or 1e-5
    for i=1,#A do
        local s = 0.0
        for j=1,#A[i] do
            assert(A[i][j] >= 0 and A[i][j] <= 1, ("%s[%d,%d] is outside [0,1]"):format(label, i, j))
            s = s + A[i][j]
        end
        T.close(s, 1.0, eps, ("%s row %d sum"):format(label, i))
    end
end

function T.build_linear_net(NN)
    local net = NN.new()
    net:add(NN.Linear(2, 2))
    NN.load_state(net, {
        { type = "Linear", W = { {1, 2}, {3, 4} }, b = { {0.5, -0.5} } },
    })
    return net
end

function T.build_classifier(NN)
    local net = NN.new()
    net:add(NN.Linear(2, 4))
    net:add(NN.Tanh())
    net:add(NN.Linear(4, 2))
    return net
end

function T.snapshot_matrix(A)
    local out = {}
    for i=1,#A do
        out[i] = {}
        for j=1,#A[i] do
            out[i][j] = A[i][j]
        end
    end
    return out
end

return T
