defmodule Tensorex.Analyzer do
  @moduledoc """
  Functions for tensor (or matrix) analysis.

  Matrices are represented as 2-rank tensors.
  """
  import Tensorex
  import Tensorex.Operator

  @doc """
  Performs the householder conversion.

  Returns a tuple of the converted vecter and the reflection matrix (`P`). The dot product of the
  reflection matrix and the given vector (`V`) (`P·V`) results to the converted vector.

      iex> Tensorex.Analyzer.householder(Tensorex.from_list([2, 3.5, -1.6, 8.2]))
      {
        %Tensorex{data: %{[0] => -9.276313923105448}, shape: [4]},
        %Tensorex{data: %{[0, 0] => -0.21560288025811625, [0, 1] => -0.3773050404517033 , [0, 2] => 0.172482304206493  , [0, 3] => -0.8839718090582762 ,
                          [1, 0] => -0.3773050404517033 , [1, 1] =>  0.8828901314218394 , [1, 2] => 0.05353593992144486, [1, 3] => -0.27437169209740486,
                          [2, 0] =>  0.172482304206493  , [2, 1] =>  0.05353593992144486, [2, 2] => 0.9755264274644824 , [2, 3] =>  0.12542705924452796,
                          [3, 0] => -0.8839718090582762 , [3, 1] => -0.27437169209740486, [3, 2] => 0.12542705924452796, [3, 3] =>  0.3571863213717944 }, shape: [4, 4]}
      }

      iex> Tensorex.Analyzer.householder(Tensorex.from_list([3.8, 0.0, 0.0, 0.0, 0.0]))
      {
        %Tensorex{data: %{[0] => 3.8}, shape: [5]},
        %Tensorex{data: %{[0, 0] => 1,
                                       [1, 1] => 1,
                                                    [2, 2] => 1,
                                                                 [3, 3] => 1,
                                                                              [4, 4] => 1}, shape: [5, 5]}
      }
  """
  @spec householder(Tensorex.t()) :: {Tensorex.t(), Tensorex.t()}
  def householder(%Tensorex{data: %{[0] => _} = store, shape: [dimension]} = vector)
      when map_size(store) === 1 do
    {vector, kronecker_delta(dimension)}
  end

  def householder(%Tensorex{shape: [dimension]} = vector) do
    dot = self_dot(vector)
    norm = if vector[[0]] < 0, do: :math.sqrt(dot), else: -:math.sqrt(dot)

    normal_vector =
      update_in(vector[[0]], &(&1 - norm))
      |> divide(dot * :math.sqrt(2 * abs(vector[[0]] - norm)))

    reflector =
      kronecker_delta(dimension)
      |> subtract(
        normal_vector
        |> multiply(normal_vector)
        |> divide(self_dot(normal_vector))
        |> multiply(2)
      )

    {%{vector | data: %{[0] => norm}}, reflector}
  end

  @spec self_dot(Tensorex.t()) :: number
  defp self_dot(%Tensorex{data: store, shape: [_]}) do
    store |> Stream.map(fn {_, value} -> value * value end) |> Enum.sum()
  end

  @doc """
  Bidiagonalizes a matrix.

  Returns a 3-element tuple containing the left-side orthogonal matrix (`U`), the bidiagonalized
  matrix (`A`) and the right-side orthogonal matrix (`V`). The dot product of them (`U·A·V`)
  results to the given matrix.

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[1, 3],
      ...>                                                     [2, 4]]))
      {
        %Tensorex{data: %{[0, 0] => -0.4472135954999581, [0, 1] => -0.8944271909999159,
                          [1, 0] => -0.8944271909999159, [1, 1] =>  0.447213595499958 }, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => -2.23606797749979  , [0, 1] => -4.919349550499538 ,
                                                         [1, 1] => -0.8944271909999157}, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] =>  1,
                                                         [1, 1] =>  1                 }, shape: [2, 2]}
      }

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[1, 3],
      ...>                                                     [2, 4],
      ...>                                                     [8, 7]]))
      {
        %Tensorex{data: %{[0, 0] => -0.12038585308576932, [0, 1] => -0.6785172735171086, [0, 2] => -0.7246527140056269 ,
                          [1, 0] => -0.24077170617153842, [1, 1] => -0.6882103774244954, [1, 2] =>  0.6843942298942036 ,
                          [2, 0] => -0.9630868246861537 , [2, 1] =>  0.2568672535457624, [2, 2] => -0.08051696822284758}, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => -8.306623862918075  , [0, 1] => -8.065852156746537 ,
                                                          [1, 1] => -2.9903225554289703}, shape: [3, 2]},
        %Tensorex{data: %{[0, 0] =>  1                  ,
                                                          [1, 1] =>  1                 }, shape: [2, 2]}
      }

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[1, 3, 4],
      ...>                                                     [2, 4, 7],
      ...>                                                     [6, 8, 9]]))
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

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[2, -4,  8],
      ...>                                                     [3,  5,  6],
      ...>                                                     [1,  9, 11],
      ...>                                                     [7, 12, 13]]))
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

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[1, 3, 5],
      ...>                                                     [2, 4, 6]]))
      {
        %Tensorex{data: %{[0, 0] => -0.4472135954999581, [0, 1] => -0.8944271909999159,
                          [1, 0] => -0.8944271909999159, [1, 1] =>  0.447213595499958 }, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => -2.23606797749979  , [0, 1] =>  9.055385138137419 ,
                                                         [1, 1] =>  1.9877674693472376, [1, 2] => -0.22086305214969304}, shape: [2, 3]},
        %Tensorex{data: %{[0, 0] =>  1                 ,
                                                         [1, 1] => -0.5432512781572743, [1, 2] => -0.8395701571521512 ,
                                                         [2, 1] => -0.8395701571521512, [2, 2] =>  0.5432512781572743 }, shape: [3, 3]}
      }

      iex> Tensorex.Analyzer.bidiagonalize(Tensorex.from_list([[1, 3, 5]]))
      {
        %Tensorex{data: %{[0, 0] => 1}, shape: [1, 1]},
        %Tensorex{data: %{[0, 0] => 1, [0, 1] => -5.830951894845301                                }, shape: [1, 3]},
        %Tensorex{data: %{[0, 0] => 1,
                                       [1, 1] => -0.5144957554275265, [1, 2] => -0.8574929257125442,
                                       [2, 1] => -0.8574929257125442, [2, 2] =>  0.5144957554275265}, shape: [3, 3]}
      }
  """
  @spec bidiagonalize(Tensorex.t()) :: {Tensorex.t(), Tensorex.t(), Tensorex.t()}
  def bidiagonalize(%Tensorex{shape: [1, columns]} = matrix) when columns in 1..2 do
    {kronecker_delta(1), matrix, kronecker_delta(columns)}
  end

  def bidiagonalize(%Tensorex{shape: [1, columns]} = matrix) do
    {householdered, reflector} = householder(matrix[[0, 1..-1]])
    right = put_in(kronecker_delta(columns)[[1..-1, 1..-1]], reflector)
    {kronecker_delta(1), put_in(matrix[[0, 1..-1]], householdered), right}
  end

  def bidiagonalize(%Tensorex{shape: [_, 1]} = matrix) do
    {householdered, reflector} = householder(matrix[[0..-1, 0]])
    {reflector, put_in(matrix[[0..-1, 0]], householdered), kronecker_delta(1)}
  end

  def bidiagonalize(%Tensorex{shape: [rows, 2]} = matrix) do
    {householdered, reflector} = householder(matrix[[0..-1, 0]])
    sub_columns = reflector |> multiply(matrix[[0..-1, 1..1]], [{1, 0}])
    {sub_left, sub_bidiagonalized, _} = sub_columns[[1..-1]] |> bidiagonalize()

    left =
      reflector |> multiply(put_in(kronecker_delta(rows)[[1..-1, 1..-1]], sub_left), [{1, 0}])

    bidiagonalized =
      matrix
      |> put_in([[0..-1, 0]], householdered)
      |> put_in([[0, 1]], sub_columns[[0, 0]])
      |> put_in([[1..-1, 1]], sub_bidiagonalized[[0..-1, 0]])

    {left, bidiagonalized, kronecker_delta(2)}
  end

  def bidiagonalize(%Tensorex{shape: [rows, columns]} = matrix) do
    {householdered_column, column_reflector} = householder(matrix[[0..-1, 0]])
    sub_columns = column_reflector |> multiply(matrix[[0..-1, 1..-1]], [{1, 0}])
    {householdered_row, row_reflector} = householder(sub_columns[[0]])

    {sub_left, sub_bidiagonalized, sub_right} =
      sub_columns[[1..-1]] |> multiply(row_reflector, [{1, 0}]) |> bidiagonalize()

    left =
      column_reflector
      |> multiply(put_in(kronecker_delta(rows)[[1..-1, 1..-1]], sub_left), [{1, 0}])

    bidiagonalized =
      matrix
      |> put_in([[0..-1, 0]], householdered_column)
      |> put_in([[0, 1..-1]], householdered_row)
      |> put_in([[1..-1, 1..-1]], sub_bidiagonalized)

    right =
      kronecker_delta(columns)
      |> put_in([[1..-1, 1..-1]], sub_right |> multiply(row_reflector, [{1, 1}]))

    {left, bidiagonalized, right}
  end

  @doc """
  Diagonalizes a square matrix.

  Returns a 2-element tuple containing the diagonalized matrix (`D`) and the square matrix (`P`)
  composed of eigen vectors of the given matrix. The dot product of (`P·D·P^-1`) results to the
  given matrix.

      iex> Tensorex.Analyzer.eigen_decomposition(Tensorex.from_list([[8, 1],
      ...>                                                           [4, 5]]))
      {
        %Tensorex{data: %{[0, 0] => 9.0               ,
                                                        [1, 1] =>  4.0                }, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => 0.7071067811865475, [0, 1] =>  0.24253562503633297,
                          [1, 0] => 0.7071067811865475, [1, 1] => -0.9701425001453319 }, shape: [2, 2]}
      }

      iex> Tensorex.Analyzer.eigen_decomposition(Tensorex.from_list([[2, 0],
      ...>                                                           [0, 3]]))
      {
        %Tensorex{data: %{[0, 0] => 2,
                                       [1, 1] => 3}, shape: [2, 2]},
        %Tensorex{data: %{[0, 0] => 1,
                                       [1, 1] => 1}, shape: [2, 2]}
      }

      iex> Tensorex.Analyzer.eigen_decomposition(Tensorex.from_list([[2, 0],
      ...>                                                           [4, 3]]))
      {
        %Tensorex{data: %{[0, 0] => 3.0,
                                         [1, 1] =>  2.0                }, shape: [2, 2]},
        %Tensorex{data: %{               [0, 1] => -0.24253562503633297,
                          [1, 0] => 1  , [1, 1] =>  0.9701425001453319 }, shape: [2, 2]}
      }

      iex> Tensorex.Analyzer.eigen_decomposition(Tensorex.from_list([[ 1,  8,  4],
      ...>                                                           [-3,  2, -6],
      ...>                                                           [ 8, -9, 11]]))
      {
        %Tensorex{data: %{[0, 0] => 15.303170410844274   ,
                                                           [1, 1] => -3.3868958657320674,
                                                                                          [2, 2] =>  2.0837254548877966 }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] =>  0.022124491408649645, [0, 1] =>  0.8133941080334768, [0, 2] =>  0.8433114989223975 ,
                          [1, 0] => -0.4151790326348706  , [1, 1] => -0.1674957147614615, [1, 2] =>  0.32735161385148664,
                          [2, 0] =>  0.909470657987536   , [2, 1] => -0.5570773829127975, [2, 2] => -0.4262236932575271 }, shape: [3, 3]}
      }

      iex> Tensorex.Analyzer.eigen_decomposition(Tensorex.from_list([[ 1,  8,  4, -8, 6],
      ...>                                                           [ 8,  2, -6, 15, 4],
      ...>                                                           [ 4, -6, 11,  7, 9],
      ...>                                                           [-8, 15,  7,  3, 2],
      ...>                                                           [ 6,  4,  9,  2, 6]]))
      {
        %Tensorex{data: %{[0, 0] => 22.48141136723747   ,
                                                          [1, 1] => -21.990125946333524  ,
                                                                                           [2, 2] => 15.981743258501801  ,
                                                                                                                           [3, 3] =>  9.870440666608177  ,
                                                                                                                                                           [4, 4] => -3.3434693460139164  }, shape: [5, 5]},
        %Tensorex{data: %{[0, 0] =>  0.22485471488273154, [0, 1] =>  0.4533959138705312  , [0, 2] =>  0.15613132580124428, [0, 3] =>  0.6959350745397415 , [0, 4] =>  0.48494317564991224 ,
                          [1, 0] =>  0.3411622703810978 , [1, 1] => -0.5996180449764692  , [1, 2] => -0.6337681498982659 , [1, 3] =>  0.31120102182464826, [1, 4] =>  0.15986982703698566 ,
                          [2, 0] =>  0.5907934013280463 , [2, 1] => -0.3000622953270929  , [2, 2] =>  0.5126463927515507 , [2, 3] => -0.38215013214304583, [2, 4] =>  0.38997529200213504 ,
                          [3, 0] =>  0.4381110287686999 , [3, 1] =>  0.5856554959326528  , [3, 2] => -0.5216463205486719 , [3, 3] => -0.4370125068120504 , [3, 4] =>  0.044402158764973956,
                          [4, 0] =>  0.5404355150079331 , [4, 1] =>  0.043134724190818236, [4, 2] =>  0.19758475025904781, [4, 3] =>  0.2860238476658191 , [4, 4] => -0.7649963886964449  }, shape: [5, 5]}
      }
  """
  @spec eigen_decomposition(Tensorex.t()) :: {Tensorex.t(), Tensorex.t()}
  def eigen_decomposition(%Tensorex{data: %{[0, 1] => divider}, shape: [2, 2]} = matrix) do
    a = (matrix[[0, 0]] + matrix[[1, 1]]) * 0.5
    b = :math.sqrt(a * a + divider * matrix[[1, 0]] - matrix[[0, 0]] * matrix[[1, 1]])

    {diagonal, square} =
      [a + b, a - b]
      |> Stream.with_index()
      |> Enum.map_reduce(%{}, fn {lambda, index}, acc ->
        x = (lambda - matrix[[0, 0]]) / divider
        norm = :math.sqrt(1 + x * x)
        store = acc |> Map.put([0, index], 1 / norm) |> Map.put([1, index], x / norm)
        {{[index, index], lambda}, store}
      end)

    {%{matrix | data: Enum.into(diagonal, %{})}, %{matrix | data: Enum.into(square, %{})}}
  end

  def eigen_decomposition(%Tensorex{data: %{[1, 0] => c}, shape: [2, 2]} = matrix) do
    a = (matrix[[0, 0]] + matrix[[1, 1]]) * 0.5
    b = :math.sqrt(a * a - matrix[[0, 0]] * matrix[[1, 1]])

    {diagonal, square} =
      [a + b, a - b]
      |> Stream.with_index()
      |> Enum.map_reduce(%{}, fn {lambda, index}, acc ->
        store =
          if lambda == matrix[[0, 0]] do
            x = (lambda - matrix[[1, 1]]) / c
            norm = :math.sqrt(x * x + 1)
            acc |> Map.put([0, index], x / norm) |> Map.put([1, index], 1 / norm)
          else
            acc |> Map.put([1, index], 1)
          end

        {{[index, index], lambda}, store}
      end)

    {%{matrix | data: Enum.into(diagonal, %{})}, %{matrix | data: square}}
  end

  def eigen_decomposition(%Tensorex{shape: [2, 2]} = matrix), do: {matrix, kronecker_delta(2)}

  def eigen_decomposition(%Tensorex{shape: [3, 3]} = matrix) do
    a = -matrix[[0, 0]] - matrix[[1, 1]] - matrix[[2, 2]]

    b =
      matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 0]] * matrix[[2, 2]] +
        matrix[[1, 1]] * matrix[[2, 2]] - matrix[[0, 1]] * matrix[[1, 0]] -
        matrix[[0, 2]] * matrix[[2, 0]] - matrix[[1, 2]] * matrix[[2, 1]]

    c =
      matrix[[0, 0]] * matrix[[1, 2]] * matrix[[2, 1]] +
        matrix[[1, 1]] * matrix[[0, 2]] * matrix[[2, 0]] +
        matrix[[2, 2]] * matrix[[0, 1]] * matrix[[1, 0]] -
        matrix[[0, 0]] * matrix[[1, 1]] * matrix[[2, 2]] -
        matrix[[0, 1]] * matrix[[1, 2]] * matrix[[2, 0]] -
        matrix[[0, 2]] * matrix[[1, 0]] * matrix[[2, 1]]

    p = (3 * b - a * a) / 9
    q = (27 * c + 2 * a * a * a - 9 * a * b) / 54
    d = :math.sqrt(-q * q - p * p * p)
    r = :math.sqrt(d * d + q * q)
    h = :math.acos(-q / r) / 3
    cbrt = :math.pow(r, 1 / 3)
    cos = :math.cos(h)
    sin = :math.sqrt(3) * :math.sin(h)

    {diagonal, square} =
      [cbrt * 2 * cos - a / 3, -cbrt * (sin + cos) - a / 3, cbrt * (sin - cos) - a / 3]
      |> Stream.with_index()
      |> Enum.map_reduce(%{}, fn {lambda, index}, acc ->
        y =
          (matrix[[0, 2]] * matrix[[1, 0]] / matrix[[1, 2]] - matrix[[0, 0]] + lambda) /
            (matrix[[0, 1]] - matrix[[0, 2]] * (matrix[[1, 1]] - lambda) / matrix[[1, 2]])

        z =
          (matrix[[0, 1]] * matrix[[1, 0]] / (matrix[[1, 1]] - lambda) - matrix[[0, 0]] + lambda) /
            (matrix[[0, 2]] - matrix[[0, 1]] * matrix[[1, 2]] / (matrix[[1, 1]] - lambda))

        norm = :math.sqrt(1 + y * y + z * z)

        store =
          acc
          |> Map.put([0, index], 1 / norm)
          |> Map.put([1, index], y / norm)
          |> Map.put([2, index], z / norm)

        {{[index, index], lambda}, store}
      end)

    {%{matrix | data: Enum.into(diagonal, %{})}, %{matrix | data: square}}
  end

  def eigen_decomposition(%Tensorex{shape: [dimension, dimension]} = matrix) do
    {left, %{data: eigens} = diagonalized, right} = singular_value_decomposition(matrix)

    signed =
      Enum.into(eigens, %{}, fn {[index | _] = indices, value} = element ->
        if left[[0, index]] * right[[0, index]] < 0, do: {indices, -value}, else: element
      end)

    {%{diagonalized | data: signed}, right}
  end

  @doc """
  Finds the singular values and the singular vectors of the given matrix.

  Returns a 3-element tuple that contains the left singular vectors (`U`), the diagonal containing
  singular values (`S`) and the right singular vectors (`V`). The dot product of them (`U·S·V^T`)
  results to the given matrix.

      iex> Tensorex.Analyzer.singular_value_decomposition(Tensorex.from_list([[1, 2, 3],
      ...>                                                                    [2, 3, 5],
      ...>                                                                    [3, 8, 9],
      ...>                                                                    [4, 5, 6]]))
      {%Tensorex{data: %{[0, 0] =>  0.2226615344045355 , [0, 1] => -0.05855885924201132, [0, 2] => -0.3843955833877411 ,
                         [1, 0] =>  0.36536351530221217, [1, 1] =>  0.15902949910885741, [1, 2] => -0.8106021820799661 ,
                         [2, 0] =>  0.7400203406817314 , [2, 1] => -0.6059441250735513 , [2, 2] =>  0.2714639225744238 ,
                         [3, 0] =>  0.518942422779177  , [3, 1] =>  0.7772465475679571 , [3, 2] =>  0.3485275837286095 }, shape: [4, 3]},
       %Tensorex{data: %{[0, 0] => 16.709361526261223  ,
                                                         [1, 1] =>  1.6718956724884724 ,
                                                                                         [2, 2] =>  1.0010006218857228 }, shape: [3, 3]},
       %Tensorex{data: %{[0, 0] =>  0.3141484053667126 , [0, 1] =>  0.9274824856259514 , [0, 2] =>  0.20269932970451207,
                         [1, 0] =>  0.6018355900828712 , [1, 1] => -0.3596812252112713 , [1, 2] =>  0.7130381046901498 ,
                         [2, 0] =>  0.73423749694166   , [2, 1] => -0.10200811285199618, [2, 2] => -0.6711852523686888 }, shape: [3, 3]}}
  """
  @spec singular_value_decomposition(Tensorex.t()) :: {Tensorex.t(), Tensorex.t(), Tensorex.t()}
  def singular_value_decomposition(%Tensorex{shape: [_, columns]} = matrix) do
    {_, bidiagonalized, right} = bidiagonalize(matrix)
    tridiagonalized = bidiagonalized |> multiply(bidiagonalized, [{0, 0}])

    eigen_values =
      tridiagonalized
      |> linearize()
      |> bisection()
      |> Stream.reject(&(&1 == 0))
      |> Enum.to_list()

    rank = length(eigen_values)

    eigen_vectors =
      eigen_values
      |> Stream.map(fn eigen_value ->
        coefficient_store =
          Enum.into(tridiagonalized.data, %{}, fn
            {[index, index] = indices, value} -> {indices, value - eigen_value}
            element -> element
          end)

        coefficient = %{tridiagonalized | data: coefficient_store}
        initial = put_in(zero([columns])[[0]], 1)
        inverse_iteration(coefficient, initial)
      end)
      |> Stream.with_index()
      |> Enum.reduce(zero([columns, rank]), fn {vector, index}, acc ->
        put_in(acc[[0..-1, index]], vector)
      end)

    store =
      eigen_values
      |> Stream.map(&:math.sqrt/1)
      |> Stream.with_index()
      |> Enum.into(%{}, fn {value, index} -> {[index, index], value} end)

    right_singular_vectors = right |> multiply(eigen_vectors, [{0, 0}])

    {decomposited_vectors, decomposited_singular_values} =
      matrix
      |> multiply(right_singular_vectors, [{1, 0}])
      |> qr_decomposition()

    left_singular_vectors =
      Enum.reduce(0..(rank - 1), decomposited_vectors, fn index, acc ->
        if decomposited_singular_values[[index, index]] < 0 do
          update_in(acc[[0..-1, index]], &negate/1)
        else
          acc
        end
      end)

    {left_singular_vectors, %Tensorex{data: store, shape: [rank, rank]}, right_singular_vectors}
  end

  @spec inverse_iteration(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  defp inverse_iteration(matrix, initial_vector) do
    inverse_iteration(lu_decomposition(matrix), initial_vector, initial_vector)
  end

  defp inverse_iteration({pivot, lower, upper} = decomposited, initial_vector, prev_difference) do
    result =
      initial_vector
      |> multiply(pivot, [{0, 0}])
      |> substitute_forward(lower)
      |> substitute_backward(upper)
      |> normalize_vector()
      |> arrange_vector()

    case result |> subtract(initial_vector) |> arrange_vector() do
      ^prev_difference -> result
      difference -> inverse_iteration(decomposited, result, difference)
    end
  end

  @spec normalize_vector(Tensorex.t()) :: Tensorex.t()
  defp normalize_vector(vector), do: vector |> divide(:math.sqrt(self_dot(vector)))
  @spec arrange_vector(Tensorex.t()) :: Tensorex.t()
  defp arrange_vector(%{data: store} = vector) when map_size(store) === 0, do: vector

  defp arrange_vector(%{data: store} = vector) do
    if elem(Enum.min(store), 1) < 0, do: negate(vector), else: vector
  end

  @doc """
  Solves a system of linear equations.

  Computes the solution vector (`X`) of the equation (`A·X = B`) where `A` is a matrix and `B` is
  a matrix or a vector.

      iex> Tensorex.Analyzer.solve(
      ...>   Tensorex.from_list([[ 3, 2, 1],
      ...>                       [ 4, 7, 6],
      ...>                       [11, 8, 9]]),
      ...>   Tensorex.from_list([6, 12, 18])
      ...> )
      %Tensorex{data: %{[0] =>  1.0000000000000002 ,
                        [1] =>  2.0000000000000004 ,
                        [2] => -1.0000000000000007}, shape: [3]}

      iex> Tensorex.Analyzer.solve(
      ...>   Tensorex.from_list([[5]]),
      ...>   Tensorex.from_list([10])
      ...> )
      %Tensorex{data: %{[0] => 2.0}, shape: [1]}
  """
  @spec solve(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  def solve(
        %Tensorex{shape: [dimension, dimension]} = coefficient,
        %Tensorex{shape: [dimension, columns]} = constant
      ) do
    {pivot, lower, upper} = lu_decomposition(coefficient)

    Enum.reduce(0..(columns - 1), constant, fn index, acc ->
      update_in(acc[[0..-1, index]], fn vector ->
        vector
        |> multiply(pivot, [{0, 0}])
        |> substitute_forward(lower)
        |> substitute_backward(upper)
      end)
    end)
  end

  def solve(
        %Tensorex{shape: [dimension, dimension]} = coefficient,
        %Tensorex{data: store, shape: [dimension]}
      ) do
    new_store = Enum.into(store, %{}, fn {[index], value} -> {[index, 0], value} end)
    solve(coefficient, %Tensorex{data: new_store, shape: [dimension, 1]})[[0..-1, 0]]
  end

  @spec substitute_forward(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  defp substitute_forward(coefficient, %{shape: [1, 1]}), do: coefficient

  defp substitute_forward(coefficient, %{shape: [dimension, dimension]} = lower) do
    substitute_forward(coefficient, lower[[1..-1]])
  end

  defp substitute_forward(coefficient, %{shape: [1 | _]} = lower) do
    update_in(coefficient[[-1]], fn x ->
      x - (lower[[0, 0..-2]] |> multiply(coefficient[[0..-2]], [{0, 0}]))
    end)
  end

  defp substitute_forward(coefficient, %{shape: [rows | _]} = lower) do
    coefficient
    |> update_in([[-rows]], fn x ->
      x - (lower[[0, 0..(-rows - 1)]] |> multiply(coefficient[[0..(-rows - 1)]], [{0, 0}]))
    end)
    |> substitute_forward(lower[[1..-1]])
  end

  @spec substitute_backward(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  defp substitute_backward(coefficient, %{shape: [1, 1]} = upper) do
    update_in(coefficient[[-1]], fn x -> x / upper[[0, 0]] end)
  end

  defp substitute_backward(coefficient, %{shape: [dimension, dimension]} = upper) do
    update_in(coefficient[[-1]], fn x -> x / upper[[-1, -1]] end)
    |> substitute_backward(upper[[0..-2]])
  end

  defp substitute_backward(coefficient, %{shape: [1 | _]} = upper) do
    update_in(coefficient[[0]], fn x ->
      (x - (upper[[0, 1..-1]] |> multiply(coefficient[[1..-1]], [{0, 0}]))) / upper[[0, 0]]
    end)
  end

  defp substitute_backward(coefficient, %{shape: [rows | _]} = upper) do
    coefficient
    |> update_in([[rows - 1]], fn x ->
      (x - (upper[[-1, rows..-1]] |> multiply(coefficient[[rows..-1]], [{0, 0}]))) /
        upper[[rows - 1, rows - 1]]
    end)
    |> substitute_backward(upper[[0..-2]])
  end

  @spec linearize(Tensorex.t()) :: [{number, number}, ...]
  defp linearize(%{shape: [dimension | _]} = tridiagonalized) do
    [{tridiagonalized[[0, 0]], 0}]
    |> Enum.concat(
      Stream.map(1..(dimension - 1), fn index ->
        {tridiagonalized[[index, index]], tridiagonalized[[index - 1, index]]}
      end)
    )
  end

  @spec bisection([{number, number}, ...]) :: Enum.t()
  defp bisection(linearized) do
    {an, bm} = List.last(linearized)

    radius =
      Stream.zip(linearized, Stream.drop(linearized, 1))
      |> Stream.map(fn {{ak, bj}, {_, bk}} -> abs(ak) + abs(bj) + abs(bk) end)
      |> Stream.concat([abs(an) + abs(bm)])
      |> Enum.max()

    narrow_down(linearized, -radius, radius)
  end

  @spec narrow_down([{number, number}, ...], number, number) :: Enum.t()
  defp narrow_down(linearized, a, b) do
    case (a + b) * 0.5 do
      c when a < c and c < b ->
        nc = n(strum(linearized, c))

        Stream.concat(
          if(nc - n(strum(linearized, b)) < 1, do: [], else: narrow_down(linearized, c, b)),
          if(n(strum(linearized, a)) - nc < 1, do: [], else: narrow_down(linearized, a, c))
        )

      c ->
        [c]
    end
  end

  @spec strum([{number, number}, ...], number) :: Enum.t()
  defp strum([{a0, _} | linearized], lambda) do
    [{1, lambda - a0} | linearized]
    |> Stream.scan(fn {ak, bj}, {pj, pk} ->
      {pk, (lambda - ak) * pk - bj * bj * pj}
    end)
    |> Stream.map(&elem(&1, 1))
  end

  @spec n(Enum.t()) :: non_neg_integer
  defp n(strum) do
    Enum.reduce(strum, {0, 1}, fn
      y, acc when y == 0 -> acc
      y, {count, prev_y} when y * prev_y < 0 -> {count + 1, y}
      y, {count, _} -> {count, y}
    end)
    |> elem(0)
  end

  @doc """
  Decomposites a square matrix into a pair of triangular matrices.

  Returns a 3-element tuple containing a row pivot matrix (`P`), a lower triangular matrix (`L`)
  and an upper triangular matrix (`U`). The dot product of them (`P·L·U`) results to the given
  matrix.

      iex> Tensorex.Analyzer.lu_decomposition(Tensorex.from_list([[10, 13, 15],
      ...>                                                        [ 5,  7,  9],
      ...>                                                        [ 9, 11, 13]]))
      {
        %Tensorex{data: %{[0, 0] =>  1  ,
                                                                         [1, 2] =>  1                 ,
                                          [2, 1] =>  1                                                }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] =>  1  ,
                          [1, 0] =>  0.9, [1, 1] =>  1                 ,
                          [2, 0] =>  0.5, [2, 1] => -0.7142857142857132, [2, 2] =>  1                 }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => 10  , [0, 1] => 13                 , [0, 2] => 15                 ,
                                          [1, 1] => -0.7000000000000011, [1, 2] => -0.5               ,
                                                                         [2, 2] =>  1.1428571428571435}, shape: [3, 3]}
      }

      iex> Tensorex.Analyzer.lu_decomposition(Tensorex.from_list([[ 0, 13, 15],
      ...>                                                        [ 5,  7,  9],
      ...>                                                        [ 9, 11, 13]]))
      {
        %Tensorex{data: %{                              [0, 1] =>  1                  ,
                                                                                        [1, 2] =>  1                 ,
                          [2, 0] => 1                                                                                }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => 1                 ,
                                                        [1, 1] =>  1                  ,
                          [2, 0] => 0.5555555555555556, [2, 1] =>  0.06837606837606834, [2, 2] =>  1                 }, shape: [3, 3]},
        %Tensorex{data: %{[0, 0] => 9                 , [0, 1] => 11                  , [0, 2] => 13                 ,
                                                        [1, 1] => 13                  , [1, 2] => 15                 ,
                                                                                        [2, 2] =>  0.7521367521367526}, shape: [3, 3]}
      }
  """
  @spec lu_decomposition(Tensorex.t()) :: {Tensorex.t(), Tensorex.t(), Tensorex.t()}
  def lu_decomposition(%Tensorex{shape: [1, 1]} = matrix) do
    delta = kronecker_delta(1)
    {delta, delta, matrix}
  end

  def lu_decomposition(%Tensorex{shape: [dimension, dimension]} = matrix) do
    [pivot_index] =
      Enum.max_by(matrix[[0..-1, 0]].data, &abs(elem(&1, 1)), fn -> {[0], nil} end) |> elem(0)

    delta = kronecker_delta(dimension)
    pivoted = matrix |> pivot_row(pivot_index)
    column = pivoted[[1..-1, 0]] |> divide(pivoted[[0, 0]])
    sub_tensor = pivoted[[1..-1, 1..-1]] |> subtract(column |> multiply(pivoted[[0, 1..-1]]))
    {sub_pivot, sub_lower, sub_upper} = lu_decomposition(sub_tensor)

    pivot =
      delta
      |> pivot_column(pivot_index)
      |> multiply(delta |> put_in([[1..-1, 1..-1]], sub_pivot), [{1, 0}])

    lower =
      delta
      |> put_in([[1..-1, 0]], sub_pivot |> multiply(column, [{1, 0}]))
      |> put_in([[1..-1, 1..-1]], sub_lower)

    upper =
      pivoted
      |> put_in([[1..-1, 0]], zero([dimension - 1]))
      |> put_in([[1..-1, 1..-1]], sub_upper)

    {pivot, lower, upper}
  end

  @spec pivot_row(Tensorex.t(), non_neg_integer) :: Tensorex.t()
  defp pivot_row(matrix, 0), do: matrix

  defp pivot_row(matrix, index) do
    matrix |> put_in([[0]], matrix[[index]]) |> put_in([[index]], matrix[[0]])
  end

  @spec pivot_column(Tensorex.t(), non_neg_integer) :: Tensorex.t()
  defp pivot_column(matrix, 0), do: matrix

  defp pivot_column(matrix, index) do
    matrix
    |> put_in([[0..-1, 0]], matrix[[0..-1, index]])
    |> put_in([[0..-1, index]], matrix[[0..-1, 0]])
  end

  @doc """
  Decomposites a matrix into a pair of an orthogonal matrix and an upper triangular matrix.

  Returns a 2-element tuple containing an orthogonal matrix (`Q`) and an upper triangular matrix
  (`R`). The dot product of them (`Q·R`) results to the given matrix.

      iex> Tensorex.Analyzer.qr_decomposition(Tensorex.from_list([[1, 2],
      ...>                                                        [3, 4],
      ...>                                                        [5, 6]]))
      {%Tensorex{data: %{[0, 0] => -0.16903085094570347, [0, 1] =>  0.89708522714506   ,
                         [1, 0] => -0.50709255283711   , [1, 1] =>  0.27602622373694213,
                         [2, 0] => -0.8451542547285165 , [2, 1] => -0.34503277967117735}, shape: [3, 2]},
       %Tensorex{data: %{[0, 0] => -5.916079783099616  , [0, 1] => -7.437357441610946  ,
                                                         [1, 1] =>  0.8280786712108249 }, shape: [2, 2]}}

      iex> Tensorex.Analyzer.qr_decomposition(Tensorex.from_list([[1, 2, 3],
      ...>                                                        [3, 4, 5]]))
      {%Tensorex{data: %{[0, 0] => -0.316227766016838 , [0, 1] => -0.9486832980505137,
                         [1, 0] => -0.9486832980505137, [1, 1] =>  0.3162277660168382}, shape: [2, 2]},
       %Tensorex{data: %{[0, 0] => -3.1622776601683795, [0, 1] => -4.42718872423573  , [0, 2] => -5.692099788303082,
                                                        [1, 1] => -0.6324555320336744, [1, 2] => -1.26491106406735}, shape: [2, 3]}}
  """
  @spec qr_decomposition(Tensorex.t()) :: {Tensorex.t(), Tensorex.t()}
  def qr_decomposition(%Tensorex{shape: [1, _]} = matrix) do
    {kronecker_delta(1), matrix}
  end

  def qr_decomposition(%Tensorex{shape: [rows, 1]} = matrix) do
    {vector, reflector} = householder(matrix[[0..-1, 0]])
    {reflector[[0..(rows - 1), 0..0]], zero([1, 1]) |> put_in([[0, 0]], vector[[0]])}
  end

  def qr_decomposition(%Tensorex{shape: [rows, columns]} = matrix) do
    diagonals = min(rows, columns)
    {vector, reflector} = householder(matrix[[0..-1, 0]])
    sub_columns = reflector |> multiply(matrix[[0..-1, 1..-1]], [{1, 0}])
    {sub_orthogonal, sub_triangular} = qr_decomposition(sub_columns[[1..-1]])

    orthogonal =
      reflector
      |> multiply(
        put_in(
          %{kronecker_delta(diagonals) | shape: [rows, diagonals]}[[1..-1, 1..-1]],
          sub_orthogonal
        ),
        [{1, 0}]
      )

    triangular =
      zero([diagonals, columns])
      |> put_in([[0]], put_in(%{vector | shape: [columns]}[[1..-1]], sub_columns[[0]]))
      |> put_in([[1..-1, 1..-1]], sub_triangular)

    {orthogonal, triangular}
  end
end
