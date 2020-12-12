defmodule Tensorex.Analyzer do
  import Tensorex
  import Tensorex.Operator

  @doc """
  Performs the householder conversion.

  Returns a tuple of the converted vecter and the convertion tensor.
  Applying the conversion tensor to the given vector results to the converted vector.

      iex> #{__MODULE__}.householder(Tensorex.from_list([2  , 3.5, -1.6,   8.2]))
      {%Tensorex{data: %{[0] => -9.276313923105448}, shape: [4]},
       %Tensorex{data: %{[0, 0] => -0.21560288025811625, [0, 1] => -0.3773050404517033 , [0, 2] => 0.172482304206493  , [0, 3] => -0.8839718090582762 ,
                         [1, 0] => -0.3773050404517033 , [1, 1] =>  0.8828901314218394 , [1, 2] => 0.05353593992144486, [1, 3] => -0.27437169209740486,
                         [2, 0] =>  0.172482304206493  , [2, 1] =>  0.05353593992144486, [2, 2] => 0.9755264274644824 , [2, 3] =>  0.12542705924452796,
                         [3, 0] => -0.8839718090582762 , [3, 1] => -0.27437169209740486, [3, 2] => 0.12542705924452796, [3, 3] =>  0.3571863213717944 }, shape: [4, 4]}}
  """
  @spec householder(Tensorex.t()) :: {Tensorex.t(), Tensorex.t()}
  def householder(%Tensorex{data: %{[0] => _} = store, shape: [dimension]} = vector)
      when map_size(store) === 1 do
    {vector, kronecker_delta(dimension)}
  end

  def householder(%Tensorex{data: %{[0] => x}, shape: [dimension]} = vector) do
    dot = self_dot(vector)
    norm = if x < 0, do: :math.sqrt(dot), else: -:math.sqrt(dot)
    normal_vector = put_in(vector[[0]], x - norm) |> divide(dot * :math.sqrt(2 * abs(x - norm)))

    tensor =
      kronecker_delta(dimension)
      |> subtract(
        normal_vector
        |> multiply(normal_vector)
        |> divide(self_dot(normal_vector))
        |> multiply(2)
      )

    {%{vector | data: %{[0] => norm}}, tensor}
  end

  @spec self_dot(Tensorex.t()) :: number
  defp self_dot(%Tensorex{data: store, shape: [_]}) do
    Enum.sum(Stream.map(store, fn {_, value} -> value * value end))
  end

  @doc """
  Bidiagonalize a 2-rank tensor.

  Returns a 3-element tuple containing a left-side orthogonal tensor (`U`), the bidiagonalized
  tensor (`A`) and a right-side orthogonal tensor (`V`). The dot product of them (`U·A·V`) equals
  to the given tensor.

      iex> #{__MODULE__}.bidiagonalize(Tensorex.from_list([[1, 3],
      ...>                                                 [2, 4]]))
      {
        %Tensorex{data: %{[0, 0] => -0.4472135954999581, [0, 1] => -0.8944271909999159,
                          [1, 0] => -0.8944271909999159, [1, 1] =>  0.447213595499958 }, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => -2.23606797749979  , [0, 1] => -4.919349550499538 ,
                                                         [1, 1] => -0.8944271909999157}, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] =>  1,
                                                         [1, 1] =>  1                 }, shape: [2, 2]}
      }

      iex> #{__MODULE__}.bidiagonalize(Tensorex.from_list([[1, 3],
      ...>                                                 [2, 4],
      ...>                                                 [8, 7]]))
      {
        %Tensorex{data: %{[0, 0] => -0.12038585308576932, [0, 1] => -0.6785172735171086, [0, 2] => -0.7246527140056269 ,
                          [1, 0] => -0.24077170617153842, [1, 1] => -0.6882103774244954, [1, 2] => 0.6843942298942036  ,
                          [2, 0] => -0.9630868246861537 , [2, 1] =>  0.2568672535457624, [2, 2] => -0.08051696822284758}, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => -8.306623862918075  , [0, 1] => -8.065852156746537 ,
                                                          [1, 1] => -2.9903225554289703}, shape: [3, 2]},
        %Tensorex{data: %{[0, 0] =>  1                  ,
                                                          [1, 1] =>  1                 }, shape: [2, 2]}
      }

      iex> #{__MODULE__}.bidiagonalize(Tensorex.from_list([[1, 3, 4],
      ...>                                                 [2, 4, 7],
      ...>                                                 [6, 8, 9]]))
      {
        %Tensorex{data: %{[0, 0] => -0.1561737618886061 , [0, 1] => -0.5866584511916395, [0, 2] => -0.7946330082138469 ,
                          [1, 0] => -0.31234752377721214, [1, 1] => -0.7338871586541569, [1, 2] =>  0.6031986925986929 ,
                          [2, 0] => -0.9370425713316364 , [2, 1] =>  0.3424054614166589, [2, 2] => -0.06862739616392308}, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => -6.4031242374328485 , [0, 1] => 14.537587950366257 ,
                                                          [1, 1] =>  4.644937425077393 , [1, 2] => -1.277069945154351  ,
                                                                                         [2, 2] =>  0.6724472155230544 }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] =>  1                  ,
                                                          [1, 1] => -0.6338226109370237, [1, 2] => -0.7734784404655207 ,
                                                          [2, 1] => -0.7734784404655207, [2, 2] =>  0.633822610937024  }, shape: [3, 3]}
      }

      iex> #{__MODULE__}.bidiagonalize(Tensorex.from_list([[2, -4,  8],
      ...>                                                      [3,  5,  6],
      ...>                                                      [1,  9, 11],
      ...>                                                      [7, 12, 13]]))
      {
        %Tensorex{data: %{[0, 0] => -0.25197631533948517, [0, 1] =>  0.11018150006547239 , [0, 2] => -0.9605551708365737 , [0, 3] =>  0.041252119101685414,
                          [1, 0] => -0.37796447300922725, [1, 1] =>  0.020828566102646506, [1, 2] =>  0.06187677855532897, [1, 3] => -0.9235151824699903  ,
                          [2, 0] => -0.12598815766974245, [2, 1] => -0.9885657272873108  , [2, 2] => -0.0793161285954799 , [2, 3] =>  0.0239528433493658  ,
                          [3, 0] => -0.881917103688197  , [3, 1] =>  0.1008167184069181  , [3, 2] =>  0.2592565906575201 , [3, 3] =>  0.3805840665510335  }, shape: [4, 4]},
        %Tensorex{data: %{[0, 0] => -7.937253933193772  , [0, 1] => 21.267756353632144   ,
                                                          [1, 1] => 11.647368945587411   , [1, 2] =>  1.3952492335365543 ,
                                                                                           [2, 2] => -8.128629398872933  }, shape: [4, 3]},
        %Tensorex{data: %{[0, 0] =>  1                  ,
                                                          [1, 1] => -0.5923904504775179  , [1, 2] => -0.8056510126494246 ,
                                                          [2, 1] => -0.8056510126494246  , [2, 2] =>  0.592390450477518  }, shape: [3, 3]}
      }

      iex> #{__MODULE__}.bidiagonalize(Tensorex.from_list([[1, 3, 5],
      ...>                                                 [2, 4, 6]]))
      {
        %Tensorex{data: %{[0, 0] => -0.4472135954999581, [0, 1] => -0.8944271909999159,
                          [1, 0] => -0.8944271909999159, [1, 1] =>  0.447213595499958 }, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => -2.23606797749979  , [0, 1] =>  9.055385138137419 ,
                                                         [1, 1] =>  1.9877674693472376, [1, 2] => -0.22086305214969304}, shape: [2, 3]},
        %Tensorex{data: %{[0, 0] =>  1                 ,
                                                         [1, 1] => -0.5432512781572743, [1, 2] => -0.8395701571521512 ,
                                                         [2, 1] => -0.8395701571521512, [2, 2] =>  0.5432512781572743 }, shape: [3, 3]}
      }
  """
  @spec bidiagonalize(Tensorex.t()) :: {Tensorex.t(), Tensorex.t(), Tensorex.t()}
  def bidiagonalize(%Tensorex{shape: [1, n]} = t) when n in 1..2 do
    {kronecker_delta(1), t, kronecker_delta(n)}
  end

  def bidiagonalize(%Tensorex{shape: [1, n]} = t) do
    {y, v} = householder(t[[0, 1..(n - 1)]])
    s = zero([1, n]) |> put_in([[0, 0]], t[[0, 0]]) |> put_in([[0, 1]], y[[0]])
    {kronecker_delta(1), s, put_in(kronecker_delta(n)[[1..-1, 1..-1]], v)}
  end

  def bidiagonalize(%Tensorex{shape: [m, 1]} = t) do
    {x, u} = householder(transpose(t, [{0, 1}])[[0]])
    {u, zero([m, 1]) |> put_in([[0, 0]], x[[0]]), kronecker_delta(1)}
  end

  def bidiagonalize(%Tensorex{shape: [m, 2]} = t) do
    {x, u} = householder(transpose(t, [{0, 1}])[[0]])
    a = u |> multiply(t[[0..(m - 1), 1..1]], [{1, 0}])
    {w, s, _} = a[[1..(m - 1)]] |> bidiagonalize()

    {u |> multiply(put_in(kronecker_delta(m)[[1..-1, 1..-1]], w), [{1, 0}]),
     put_in(zero([m, 2])[[0, 0]], x[[0]])
     |> put_in([[0, 1]], a[[0, 0]])
     |> put_in([[1..-1, 1..-1]], s), kronecker_delta(2)}
  end

  def bidiagonalize(%Tensorex{shape: [m, n]} = t) do
    {x, u} = householder(transpose(t, [{0, 1}])[[0]])
    a = u |> multiply(t[[0..(m - 1), 1..(n - 1)]], [{1, 0}])
    {y, v} = householder(a[[0]])
    {w, s, z} = a[[1..(m - 1)]] |> multiply(v, [{1, 0}]) |> bidiagonalize()

    {u |> multiply(put_in(kronecker_delta(m)[[1..-1, 1..-1]], w), [{1, 0}]),
     zero([m, n])
     |> put_in([[0, 0]], x[[0]])
     |> put_in([[0, 1]], y[[0]])
     |> put_in([[1..-1, 1..-1]], s),
     put_in(kronecker_delta(n)[[1..-1, 1..-1]], z |> multiply(transpose(v, [{0, 1}]), [{1, 0}]))}
  end

  @doc """
  Returns pairs of an eigen value and a corresponding eigen vector of a 2-rank symmetric tensor.

  You can also pass a non symmetric tensor only if the tensor has 2 or 3 dimensions.

      iex> #{__MODULE__}.eigens(Tensorex.from_list([[8, 1],
      ...>                                          [4, 5]]))
      ...> |> Enum.to_list()
      [{4.0, %Tensorex{data: %{[0] => 0.24253562503633297, [1] => -0.9701425001453319}, shape: [2]}},
       {9.0, %Tensorex{data: %{[0] => 0.7071067811865475 , [1] =>  0.7071067811865475}, shape: [2]}}]

      iex> #{__MODULE__}.eigens(Tensorex.from_list([[2, 0],
      ...>                                          [0, 3]]))
      ...> |> Enum.to_list()
      [{2.0, %Tensorex{data: %{[0] => -1.0         }, shape: [2]}},
       {3.0, %Tensorex{data: %{            [1] => 1}, shape: [2]}}]

      iex> #{__MODULE__}.eigens(Tensorex.from_list([[2, 0],
      ...>                                          [4, 3]]))
      ...> |> Enum.to_list()
      [{2.0, %Tensorex{data: %{[0] => -0.24253562503633297, [1] => 0.9701425001453319}, shape: [2]}},
       {3.0, %Tensorex{data: %{                             [1] => 1                 }, shape: [2]}}]

      iex> #{__MODULE__}.eigens(Tensorex.from_list([[ 1,  8,  4],
      ...>                                          [-3,  2, -6],
      ...>                                          [ 8, -9, 11]]))
      ...> |> Enum.to_list()
      [{15.303170410844274 , %Tensorex{data: %{[0] => 0.022124491408649645, [1] => -0.4151790326348706, [2] =>  0.909470657987536 }, shape: [3]}},
       {-3.3868958657320674, %Tensorex{data: %{[0] => 0.8133941080334768  , [1] => -0.1674957147614615, [2] => -0.5570773829127975}, shape: [3]}},
       { 2.0837254548877966, %Tensorex{data: %{[0] => 0.8433114989223975  , [1] => 0.32735161385148664, [2] => -0.4262236932575271}, shape: [3]}}]

      iex> #{__MODULE__}.eigens(Tensorex.from_list([[ 1,  8,  4, -8, 6],
      ...>                                          [ 8,  2, -6, 15, 4],
      ...>                                          [ 4, -6, 11,  7, 9],
      ...>                                          [-8, 15,  7,  3, 2],
      ...>                                          [ 6,  4,  9,  2, 6]]))
      ...> |> Enum.to_list()
      [{ 22.4814 , %Tensorex{data: %{[0] =>  0.416062, [1] =>   0.631273, [2] =>  1.09318 , [3] =>  0.810663 , [4] => 1}, shape: [5]}},
       {-21.9901 , %Tensorex{data: %{[0] => 10.5112  , [1] => -13.9011  , [2] => -6.9564  , [3] => 13.5774   , [4] => 1}, shape: [5]}},
       { 15.9817 , %Tensorex{data: %{[0] =>  0.790199, [1] => -3.20758  , [2] =>  2.59456 , [3] => -2.64011  , [4] => 1}, shape: [5]}},
       {  9.87044, %Tensorex{data: %{[0] =>  2.43314 , [1] =>  1.08802  , [2] => -1.33608 , [3] => -1.52789  , [4] => 1}, shape: [5]}},
       { -3.34347, %Tensorex{data: %{[0] => -0.633916, [1] => -0.208981 , [2] => -0.509774, [3] => -0.0580423, [4] => 1}, shape: [5]}}]
  """
  @spec eigens(Tensorex.t()) :: Enum.t()
  def eigens(%Tensorex{shape: [2, 2]} = t) do
    a = (t[[0, 0]] + t[[1, 1]]) * 0.5
    b = :math.sqrt(a * a + t[[0, 1]] * t[[1, 0]] - t[[0, 0]] * t[[1, 1]])

    Stream.map([a - b, a + b], fn l ->
      cond do
        t[[0, 1]] == 0 and l == t[[0, 0]] ->
          {l,
           zero([2]) |> put_in([[0]], l - t[[1, 1]]) |> put_in([[1]], t[[1, 0]]) |> normalize(2)}

        t[[0, 1]] == 0 ->
          {l, zero([2]) |> put_in([[1]], 1)}

        true ->
          {l,
           zero([2])
           |> put_in([[0]], 1)
           |> put_in([[1]], (l - t[[0, 0]]) / t[[0, 1]])
           |> normalize(2)}
      end
    end)
  end

  def eigens(%Tensorex{shape: [3, 3]} = t) do
    a = -t[[0, 0]] - t[[1, 1]] - t[[2, 2]]

    b =
      t[[0, 0]] * t[[1, 1]] + t[[0, 0]] * t[[2, 2]] + t[[1, 1]] * t[[2, 2]] -
        t[[0, 1]] * t[[1, 0]] - t[[0, 2]] * t[[2, 0]] - t[[1, 2]] * t[[2, 1]]

    c =
      t[[0, 0]] * t[[1, 2]] * t[[2, 1]] + t[[1, 1]] * t[[0, 2]] * t[[2, 0]] +
        t[[2, 2]] * t[[0, 1]] * t[[1, 0]] -
        t[[0, 0]] * t[[1, 1]] * t[[2, 2]] - t[[0, 1]] * t[[1, 2]] * t[[2, 0]] -
        t[[0, 2]] * t[[1, 0]] * t[[2, 1]]

    p = (3 * b - a * a) / 9
    q = (27 * c + 2 * a * a * a - 9 * a * b) / 54

    case q * q + p * p * p do
      u when u < 0 ->
        d = :math.sqrt(-u)
        r = :math.sqrt(d * d + q * q)
        h = :math.acos(-q / r)

        [
          :math.pow(r, 1 / 3) * 2 * :math.cos(h / 3) - a / 3,
          :math.pow(r, 1 / 3) * (:math.sqrt(3) * :math.sin(-h / 3) - :math.cos(h / 3)) - a / 3,
          :math.pow(r, 1 / 3) * (:math.sqrt(3) * :math.sin(h / 3) - :math.cos(h / 3)) - a / 3
        ]

      u ->
        d = :math.sqrt(u)
        [:math.pow(d - q, 1 / 3) + :math.pow(-q - d, 1 / 3) - a / 3]
    end
    |> Stream.map(fn l ->
      y =
        (t[[0, 2]] * t[[1, 0]] / t[[1, 2]] - t[[0, 0]] + l) /
          (t[[0, 1]] - t[[0, 2]] * (t[[1, 1]] - l) / t[[1, 2]])

      z =
        (t[[0, 1]] * t[[1, 0]] / (t[[1, 1]] - l) - t[[0, 0]] + l) /
          (t[[0, 2]] - t[[0, 1]] * t[[1, 2]] / (t[[1, 1]] - l))

      {l, zero([3]) |> put_in([[0]], 1) |> put_in([[1]], y) |> put_in([[2]], z) |> normalize(2)}
    end)
  end

  defp norm(%Tensorex{data: store, shape: [_]}, p) do
    Stream.map(store, fn {_, value} -> :math.pow(value, p) end)
    |> Enum.sum()
    |> :math.pow(1 / p)
  end

  defp normalize(%Tensorex{shape: [_]} = vector, p) do
    vector |> divide(norm(vector, p))
  end

  @doc """
  Decomposites a 2-rank tensor into a dot producy of 2 orthogonal tensor and a diagonal one.

  Returns a 3-element tuple that contains the left singular vector (`U`), the diagonal containing
  singular values (`S`) and the right singular vector (`V`). The dot product of them (`U·S·V^T`)
  is equal to the given tensor.

      iex> #{__MODULE__}.singular_value_decomposition(Tensorex.from_list([[1, 2, 3],
      ...>                                                                [4, 5, 6],
      ...>                                                                [7, 8, 9],
      ...>                                                                [0, 1, 2]]))
      {%Tensorex{data: %{[0, 0] => 0.213906, [0, 1] =>  0.581014 , [0, 2] => -0.784465, [0, 3] =>  0.0358057,
                         [1, 0] => 0.517391, [1, 1] =>  0.125058 , [1, 2] =>  0.196116, [1, 3] => -0.823532 ,
                         [2, 0] => 0.820877, [2, 1] => -0.330899 ,                      [2, 3] =>  0.465475 ,
                         [3, 0] => 0.112744, [3, 1] =>  0.733    , [3, 2] =>  0.588348, [3, 3] =>  0.322252 }, shape: [4, 4]},
       %Tensorex{data: %{[0, 0] => 16.9557,
                                             [1, 1] =>  1.58253}, shape: [4, 3]},
       %Tensorex{data: %{[0, 0] => 0.473564, [0, 1] => -0.78043  , [0, 2] =>  0.408248,
                         [1, 0] => 0.571757, [1, 1] => -0.0801737, [1, 2] => -0.816497,
                         [2, 0] => 0.669949, [2, 1] =>  0.620082 , [2, 2] =>  0.408248}, shape: [3, 3]}}
  """
  @spec singular_value_decomposition(Tensorex.t(), number) ::
          {Tensorex.t(), Tensorex.t(), Tensorex.t()}
  def singular_value_decomposition(%Tensorex{shape: [_, n]} = t, delta \\ 1.0e-15)
      when delta > 0 do
    {uu, b, vv} = bidiagonalize(t)
    {q, ss} = b |> multiply(b, [{0, 0}]) |> schur_decomposition(delta * 1.0e-3)

    sign =
      Enum.reduce(0..(n - 1), kronecker_delta(n), fn i, acc ->
        if ss[[i, i]] < 0, do: put_in(acc[[i, i]], -1), else: acc
      end)

    s =
      Enum.reduce(0..(n - 1), ss |> multiply(sign, [{1, 0}]), fn i, acc ->
        update_in(acc[[i, i]], &:math.sqrt/1)
      end)

    v = vv |> multiply(q, [{0, 0}])
    {u, _} = b |> multiply(q, [{1, 0}]) |> multiply(sign, [{1, 0}]) |> qr_decomposition({1, 0})
    {uu |> multiply(u, [{1, 0}]), s, v}
  end

  defp w_to_u(%Tensorex{data: d, shape: [_, _]} = t, delta) do
    Stream.map(d, fn {i, v} ->
      Stream.map(v, fn {j, u} ->
        nil
      end)
    end)
  end

  defp u_to_v(%Tensorex{shape: [_]} = t) do
  end

  @doc """
  Decomposites a 2-rank tensor into a dot product of an orthogonal tensor and an upper triangular
  tensor.

  Returns a two-element tuple containing the orthogonal tensor (`Q`) and the schur form of the
  given tensor(`U`). The dot product of `Q·U·Q^T` is equal to the given tensor.

      iex> #{__MODULE__}.schur_decomposition(Tensorex.from_list([[1, 2],
      ...>                                                       [4, 5]]))
      {%Tensorex{data: %{[0, 0] => -0.8068982213550735, [0, 1] => 0.5906904945688723,
                         [1, 0] =>  0.5906904945688723, [1, 1] => 0.8068982213550735}, shape: [2, 2]},
       %Tensorex{data: %{[0, 0] => -0.4641016151377553, [0, 1] => 2.0000000000000004,
                                                        [1, 1] => 6.464101615137754 }, shape: [2, 2]}}
  """
  @spec schur_decomposition(Tensorex.t(), number) :: {Tensorex.t(), Tensorex.t()}
  def schur_decomposition(%Tensorex{shape: [n, n]} = t, d \\ 1.0e-12)
      when is_number(d) and d > 0 do
    e =
      eigens(t[[(n - 2)..(n - 1), (n - 2)..(n - 1)]])
      |> Stream.map(&elem(&1, 0))
      |> Enum.min_by(&abs(&1 - t[[n - 1, n - 1]]))
      |> multiply(kronecker_delta(n))

    {q, r} = qr_decomposition(t |> subtract(e), {1, 0})
    u = r |> multiply(q, [{1, 0}]) |> add(e)

    if Enum.all?(q |> multiply(u, [{1, 0}]) |> multiply(q, [{1, 1}]) |> errors(t), &(&1 <= d)) do
      {q, u}
    else
      {p, a} = schur_decomposition(u, d)
      {q |> multiply(p, [{1, 0}]), a}
    end
  end

  @spec errors(Tensorex.t(), Tensorex.t()) :: Enum.t()
  defp errors(t1, t2) do
    %Tensorex{data: d} = t1 |> subtract(t2)

    Stream.map(d, fn
      {i, v} -> [abs(v) / max(max(abs(t1[i]), abs(t2[i])), 1)]
    end)
  end

  @doc """
  Decomposites a 2-rank tensor into a dot product of an orthogonal tensor and an upper triangular
  tensor.

      iex> #{__MODULE__}.qr_decomposition(Tensorex.from_list([[1, 2],
      ...>                                                    [3, 4],
      ...>                                                    [5, 6]]), {1, 0})
      {%Tensorex{data: %{[0, 0] => -0.16903085094570347, [0, 1] =>  0.89708522714506   ,
                         [1, 0] => -0.50709255283711   , [1, 1] =>  0.27602622373694213,
                         [2, 0] => -0.8451542547285165 , [2, 1] => -0.34503277967117735}, shape: [3, 2]},
       %Tensorex{data: %{[0, 0] => -5.916079783099616  , [0, 1] => -7.437357441610946  ,
                                                         [1, 1] =>  0.8280786712108249 }, shape: [2, 2]}}

      iex> #{__MODULE__}.qr_decomposition(Tensorex.from_list([[1, 2, 3],
      ...>                                                    [3, 4, 5]]), {1, 0})
      {%Tensorex{data: %{[0, 0] => -0.316227766016838 , [0, 1] => -0.9486832980505137,
                         [1, 0] => -0.9486832980505137, [1, 1] =>  0.3162277660168382}, shape: [2, 2]},
       %Tensorex{data: %{[0, 0] => -3.1622776601683795, [0, 1] => -4.42718872423573  , [0, 2] => -5.692099788303082,
                                                        [1, 1] => -0.6324555320336744, [1, 2] => -1.26491106406735}, shape: [2, 3]}}
  """
  @spec qr_decomposition(Tensorex.t(), {0 | 1, 0 | 1}) :: {Tensorex.t(), Tensorex.t()}
  def qr_decomposition(%Tensorex{shape: [1, _]} = t, {a1, a2})
      when is_integer(a1) and is_integer(a2) and a1 in 0..1 and a2 in 0..1 do
    {kronecker_delta(1), t}
  end

  def qr_decomposition(%Tensorex{shape: [m, 1]} = t, {a1, a2})
      when is_integer(a1) and is_integer(a2) and a1 in 0..1 and a2 in 0..1 do
    {x, q} = householder(transpose(t, [{0, 1}])[[0]])

    {q[[0..(m - 1), 0..0]] |> transpose(if a1 === 0, do: [{0, 1}], else: []),
     zero([1, 1]) |> put_in([[0, 0]], x[[0]])}
  end

  def qr_decomposition(%Tensorex{shape: [m, n]} = t, {a1, a2})
      when is_integer(a1) and is_integer(a2) and a1 in 0..1 and a2 in 0..1 do
    {x, u} = householder(transpose(t, [{0, 1}])[[0]])
    a = u |> multiply(t[[0..(m - 1), 1..(n - 1)]], [{1, 0}])
    {w, r} = a[[1..(m - 1)]] |> qr_decomposition({1, 0})

    q =
      zero([m, min(m, n)]) |> put_in([[0..min(m, n)-1, 0..min(m, n)-1]], kronecker_delta(min(m, n))) |> put_in([[1..-1, 1..-1]], w)

    {u |> multiply(q, [{1, 0}]) |> transpose(if a1 === 0, do: [{0, 1}], else: []),
     zero([min(m, n), n])
     |> put_in([[0, 0]], x[[0]])
     |> put_in([[0, 1..-1]], a[[0]])
     |> put_in([[1..-1, 1..-1]], r)
     |> transpose(if a2 === 1, do: [{0, 1}], else: [])}
  end
end
