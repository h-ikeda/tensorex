defmodule Tensorex.Operator do
  @moduledoc """
  Functions for basic arithmetic operations with tensors.
  """
  @doc """
  Adds two tensors.

      iex> Tensorex.Operator.add(
      ...>   Tensorex.from_list([[0,  1  ,  2  ],
      ...>                       [3, -4  , -5.5]]),
      ...>   Tensorex.from_list([[3, -2  , -2  ],
      ...>                       [6, -8.1, 12  ]]))
      %Tensorex{data: %{[0, 0] => 3, [0, 1] =>  -1,
                        [1, 0] => 9, [1, 1] => -12.1, [1, 2] => 6.5}, shape: [2, 3]}

      iex> Tensorex.Operator.add(
      ...>   Tensorex.from_list([[0  ,  1  ,  2  ],
      ...>                       [3  , -4  , -5.5]]),
      ...>   Tensorex.from_list([[0.0, -1  , -2  ],
      ...>                       [6  , -8.1, 12  ]]))
      %Tensorex{data: %{[1, 0] => 9, [1, 1] => -12.1, [1, 2] => 6.5}, shape: [2, 3]}
  """
  @spec add(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  def add(%Tensorex{data: left, shape: shape} = tensor, %Tensorex{data: right, shape: shape}) do
    {small_store, large_store} =
      if map_size(left) < map_size(right), do: {left, right}, else: {right, left}

    new_store =
      Enum.reduce(small_store, large_store, fn {index, value2}, acc ->
        case Map.fetch(acc, index) do
          {:ok, value1} when value1 + value2 == 0 -> Map.delete(acc, index)
          {:ok, value1} -> Map.put(acc, index, value1 + value2)
          :error -> Map.put(acc, index, value2)
        end
      end)

    %{tensor | data: new_store}
  end

  @doc """
  Subtracts a tensor from another.

      iex> Tensorex.Operator.subtract(
      ...>   Tensorex.from_list([[0,  1,  2], [3, -4,   -5.5]]),
      ...>   Tensorex.from_list([[3, -2, -2], [6, -8.1, 12  ]]))
      %Tensorex{data: %{[0, 0] => -3, [0, 1] => 3  , [0, 2] =>   4  ,
                        [1, 0] => -3, [1, 1] => 4.1, [1, 2] => -17.5}, shape: [2, 3]}

      iex> Tensorex.Operator.subtract(
      ...>   Tensorex.from_list([[0,   1, 2], [3, -4,   -5.5]]),
      ...>   Tensorex.from_list([[0.0, 1, 2], [6, -8.1, 12  ]]))
      %Tensorex{data: %{[1, 0] => -3, [1, 1] => 4.1, [1, 2] => -17.5}, shape: [2, 3]}
  """
  @spec subtract(Tensorex.t(), Tensorex.t()) :: Tensorex.t()
  def subtract(%Tensorex{data: left, shape: shape} = tensor, %Tensorex{data: right, shape: shape}) do
    new_store =
      Enum.reduce(right, left, fn {index, value2}, acc ->
        case Map.fetch(acc, index) do
          {:ok, value1} when value1 - value2 == 0 -> Map.delete(acc, index)
          {:ok, value1} -> Map.put(acc, index, value1 - value2)
          :error -> Map.put(acc, index, -value2)
        end
      end)

    %{tensor | data: new_store}
  end

  @doc """
  Negates a tensor.

      iex> Tensorex.Operator.negate(
      ...>   Tensorex.from_list([[ 2  , 3.5, -4  , 0  ],
      ...>                       [-2.2, 6  ,  0.0, 5.5]]))
      %Tensorex{data: %{[0, 0] => -2  , [0, 1] => -3.5, [0, 2] => 4,
                        [1, 0] =>  2.2, [1, 1] => -6  ,              [1, 3] => -5.5}, shape: [2, 4]}
  """
  @spec negate(Tensorex.t()) :: Tensorex.t()
  def negate(%Tensorex{data: store} = tensor) do
    %{tensor | data: Enum.into(store, %{}, fn {index, value} -> {index, -value} end)}
  end

  @doc """
  Makes a product of tensors.

  If both arguments are tensors, it returns a tensor product of them. When one of arguments is a
  `t:number/0`, then all elements of the tensor will be amplified by the scalar.

      iex> Tensorex.Operator.multiply(
      ...>   Tensorex.from_list([2, 5.2, -4  , 0  ]),
      ...>   Tensorex.from_list([2, 3.5, -1.6, 8.2]))
      %Tensorex{data: %{[0, 0] => 4   , [0, 1] =>   7.0, [0, 2] => -3.2 , [0, 3] => 16.4 ,
                        [1, 0] => 10.4, [1, 1] =>  18.2, [1, 2] => -8.32, [1, 3] => 42.64,
                        [2, 0] => -8  , [2, 1] => -14.0, [2, 2] =>  6.4 , [2, 3] => -32.8}, shape: [4, 4]}

      iex> Tensorex.Operator.multiply(3.5,
      ...>   Tensorex.from_list([[2   ,  3.5, -1.5, 8.0],
      ...>                       [4.12, -2  ,  1  , 0  ]]))
      %Tensorex{data: %{[0, 0] =>  7.0 , [0, 1] => 12.25, [0, 2] => -5.25, [0, 3] => 28.0,
                        [1, 0] => 14.42, [1, 1] => -7.0 , [1, 2] =>  3.5                 }, shape: [2, 4]}
  """
  @spec multiply(Tensorex.t() | number, Tensorex.t() | number) :: Tensorex.t()
  def multiply(
        %Tensorex{data: left_store, shape: left_shape},
        %Tensorex{data: right_store, shape: right_shape}
      ) do
    new_store =
      Stream.map(left_store, fn {left_index, left_value} ->
        Stream.map(right_store, fn {right_index, right_value} ->
          {left_index ++ right_index, left_value * right_value}
        end)
      end)
      |> Stream.concat()
      |> Enum.into(%{})

    %Tensorex{data: new_store, shape: left_shape ++ right_shape}
  end

  def multiply(scalar, %Tensorex{data: store} = tensor) when is_number(scalar) do
    %{tensor | data: Enum.into(store, %{}, fn {index, value} -> {index, scalar * value} end)}
  end

  def multiply(%Tensorex{data: store} = tensor, scalar) when is_number(scalar) do
    %{tensor | data: Enum.into(store, %{}, fn {index, value} -> {index, scalar * value} end)}
  end

  @doc """
  Makes a dot product of tensors.

  Components specified by the `axes` will be sumed up (or contracted).

      iex> Tensorex.Operator.multiply(
      ...>   Tensorex.from_list([0, 0.0,  0.0, 0  ]),
      ...>   Tensorex.from_list([2, 3.5, -1.6, 8.2]), [{0, 0}])
      0.0

      iex> Tensorex.Operator.multiply(
      ...>   Tensorex.from_list([[2  , 3.5, -1.6,   8.2],
      ...>                       [1.1, 3.0,  8  , -12.1]]),
      ...>   Tensorex.from_list([[0  , 0.0],
      ...>                       [0.0, 0  ],
      ...>                       [0.0, 0  ],
      ...>                       [0  , 0  ]]), [{0, 1}, {1, 0}])
      0.0

      iex> Tensorex.Operator.multiply(
      ...>   Tensorex.from_list([2, 5.2, -4  , 0  ]),
      ...>   Tensorex.from_list([2, 3.5, -1.6, 8.2]), [{0, 0}])
      28.6

      iex> Tensorex.Operator.multiply(
      ...>   Tensorex.from_list([[ 2   ,  5.5, -4  , 0  ],
      ...>                       [ 4.12, -2  ,  1  , 0  ]]),
      ...>   Tensorex.from_list([[ 2   ,  3.5],
      ...>                       [-1.6 ,  8.2],
      ...>                       [ 2   , -3.5],
      ...>                       [-1.5 ,  8.0]]), [{0, 1}])
      %Tensorex{data: %{[0, 0] => 18.42, [0, 1] =>  30.584, [0, 2] => -10.42, [0, 3] =>  29.96,
                        [1, 0] =>  4.0 , [1, 1] => -25.2  , [1, 2] =>  18.0 , [1, 3] => -24.25,
                        [2, 0] => -4.5 , [2, 1] =>  14.6  , [2, 2] => -11.5 , [2, 3] =>  14.0 }, shape: [4, 4]}
  """
  @spec multiply(Tensorex.t(), Tensorex.t(), [{non_neg_integer, non_neg_integer}]) ::
          Tensorex.t() | number
  def multiply(
        %Tensorex{data: left, shape: left_shape},
        %Tensorex{data: right, shape: right_shape},
        axes
      )
      when is_list(axes) do
    shape =
      Enum.reduce(axes, [left_shape, right_shape], fn
        {left_axis, right_axis}, [left_acc, right_acc] ->
          [List.replace_at(left_acc, left_axis, nil), List.replace_at(right_acc, right_axis, nil)]
      end)
      |> Stream.concat()
      |> Enum.filter(& &1)

    {left_axes, right_axes} = Enum.unzip(axes)
    left_group = group_by_contraction(left, left_axes)
    right_group = group_by_contraction(right, right_axes)
    store = multiply_with_contraction(left_group, right_group, length(axes))

    cond do
      Enum.empty?(store) and shape == [] ->
        0.0

      shape == [] ->
        Enum.fetch!(store, 0) |> elem(1)

      true ->
        %Tensorex{data: store |> Enum.into(%{}), shape: shape}
    end
  end

  @typep contraction_map ::
           %{non_neg_integer => contraction_map} | %{optional([non_neg_integer, ...]) => number}
  @spec group_by_contraction(Enum.t(), [{non_neg_integer, non_neg_integer}]) :: contraction_map
  defp group_by_contraction(elements, []) do
    Enum.into(elements, %{}, fn {index, value} -> {Enum.filter(index, & &1), value} end)
  end

  defp group_by_contraction(store, [axis | axes]) do
    Enum.group_by(
      store,
      fn {index, _} -> Enum.fetch!(index, axis) end,
      fn {index, value} -> {List.replace_at(index, axis, nil), value} end
    )
    |> Enum.into(%{}, fn {grouped_axis, elements} ->
      {grouped_axis, group_by_contraction(elements, axes)}
    end)
  end

  @spec multiply_with_contraction(Enum.t(), Enum.t(), non_neg_integer) :: Enum.t()
  defp multiply_with_contraction(left, right, 0) do
    Stream.map(left, fn {left_index, left_value} ->
      Stream.map(right, fn {right_index, right_value} ->
        {left_index ++ right_index, left_value * right_value}
      end)
    end)
    |> Stream.concat()
  end

  defp multiply_with_contraction(left, right, depth) do
    Stream.map(left, fn {contract_index, left_elements} ->
      case Map.fetch(right, contract_index) do
        {:ok, right_elements} ->
          multiply_with_contraction(left_elements, right_elements, depth - 1)

        :error ->
          []
      end
    end)
    |> Stream.concat()
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Stream.flat_map(fn {index, values} ->
      case Enum.sum(values) do
        value when value == 0 -> []
        value -> [{index, value}]
      end
    end)
  end

  @doc """
  Returns a transposed tensor.

      iex> Tensorex.Operator.transpose(
      ...>   Tensorex.from_list([[[ 2   ,  5.5, -4, 0  ],
      ...>                        [ 4.12, -2  ,  1, 0  ]],
      ...>                       [[ 3   ,  1.2,  5, 8.9],
      ...>                        [ 1   ,  6  ,  7, 1.3]]]), [{0, 2}])
      %Tensorex{data: %{[0, 0, 0] =>  2   , [0, 0, 1] => 3  ,
                        [0, 1, 0] =>  4.12, [0, 1, 1] => 1  ,
                        [1, 0, 0] =>  5.5 , [1, 0, 1] => 1.2,
                        [1, 1, 0] => -2   , [1, 1, 1] => 6  ,
                        [2, 0, 0] => -4   , [2, 0, 1] => 5  ,
                        [2, 1, 0] =>  1   , [2, 1, 1] => 7  ,
                                            [3, 0, 1] => 8.9,
                                            [3, 1, 1] => 1.3}, shape: [4, 2, 2]}
  """
  @spec transpose(Tensorex.t(), [{non_neg_integer, non_neg_integer}, ...]) :: Tensorex.t()
  def transpose(%Tensorex{data: store, shape: shape}, axes) when is_list(axes) do
    new_store =
      Enum.into(store, %{}, fn {index, value} ->
        new_index =
          Enum.reduce(axes, index, fn {left_axis, right_axis}, acc ->
            acc
            |> List.replace_at(left_axis, Enum.fetch!(acc, right_axis))
            |> List.replace_at(right_axis, Enum.fetch!(acc, left_axis))
          end)

        {new_index, value}
      end)

    new_shape =
      Enum.reduce(axes, shape, fn {left_axis, right_axis}, acc ->
        acc
        |> List.replace_at(left_axis, Enum.fetch!(acc, right_axis))
        |> List.replace_at(right_axis, Enum.fetch!(acc, left_axis))
      end)

    %Tensorex{data: new_store, shape: new_shape}
  end

  @doc """
  Divides all elements of the tensor by the scalar.

      iex> Tensorex.Operator.divide(
      ...>   Tensorex.from_list([[2  , 3.5, -1.6,   8.2],
      ...>                       [1.1, 3.0,  0  , -12.1]]), 4)
      %Tensorex{data: %{[0, 0] => 0.5  , [0, 1] => 0.875, [0, 2] => -0.4, [0, 3] =>  2.05 ,
                        [1, 0] => 0.275, [1, 1] => 0.75 ,                 [1, 3] => -3.025}, shape: [2, 4]}
  """
  @spec divide(Tensorex.t(), number) :: Tensorex.t()
  def divide(%Tensorex{data: store} = tensor, scalar) when is_number(scalar) do
    %{tensor | data: Enum.into(store, %{}, fn {index, value} -> {index, value / scalar} end)}
  end
end
