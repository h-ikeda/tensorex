defmodule Tensorex do
  @moduledoc """
  Functions to operate basic commands with tensors.
  """
  @typedoc """
  Represents a tensor.

  The data structure is a map with list of indices keys and numeric values. Zero values are
  omitted. The shape is a list of each dimension at the order.

  This module implements the `Access` behaviour. So that you can access elements of the tensor via
  `tensor[indices]` syntax, where `indices` must be a list of `t:integer/0`s or `t:Range.t/0`s. See
  `fetch/2` for concrete examples.
  """
  @type t :: %Tensorex{
          data: %{optional([non_neg_integer, ...]) => number},
          shape: [pos_integer, ...]
        }
  defstruct [:data, :shape]

  @doc """
  Creates a new tensor from a list (of lists (of lists of ...)).

      iex> Tensorex.from_list([1.1, 2.1, -5.3, 4])
      %Tensorex{data: %{[0] => 1.1, [1] => 2.1, [2] => -5.3, [3] => 4}, shape: [4]}

      iex> Tensorex.from_list([[1.1,  2.1, -5.3, 4  ],
      ...>                     [0.8, -8,   21.4, 3.3]])
      %Tensorex{data: %{[0, 0] => 1.1, [0, 1] =>  2.1, [0, 2] => -5.3, [0, 3] => 4  ,
                        [1, 0] => 0.8, [1, 1] => -8,   [1, 2] => 21.4, [1, 3] => 3.3}, shape: [2, 4]}

      iex> Tensorex.from_list([[[0.0, 0.0, 0.0],
      ...>                      [0.0, 0.0, 0.0]],
      ...>                     [[0.0, 0.0, 0.0],
      ...>                      [0.0, 0.0, 0.0]]])
      %Tensorex{data: %{}, shape: [2, 2, 3]}
  """
  @spec from_list(Enum.t()) :: t
  def from_list(data), do: %Tensorex{data: build(data) |> Enum.into(%{}), shape: count(data)}
  @spec count(Enum.t()) :: [pos_integer, ...]
  defp count(data) do
    [nested_shape] =
      data
      |> Stream.map(fn
        value when is_number(value) -> []
        nested_data -> count(nested_data)
      end)
      |> Enum.uniq()

    [Enum.count(data) | nested_shape]
  end

  @spec build(Enum.t()) :: Enum.t()
  defp build(data) do
    data
    |> Stream.with_index()
    |> Stream.map(fn
      {value, _} when value == 0 ->
        []

      {value, index} when is_number(value) ->
        [{[index], value}]

      {nested_data, index} ->
        build(nested_data)
        |> Stream.map(fn {nested_indices, value} -> {[index | nested_indices], value} end)
    end)
    |> Stream.concat()
  end

  defguardp is_indices(indices, shape) when is_list(indices) and length(indices) <= length(shape)
  @behaviour Access
  @doc """
  Returns a tensor or a number stored at the index.

  The key can be a list of indices or ranges. If integer indices are given, it returns a tensor
  or a numeric value specified by the index. If ranges are given, it returns a tensor consisting
  partial elements.

  Negative indices are counted from the end.

      iex> Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                      [ 4   ,    5  ,  -6.1],
      ...>                      [ 0.9 ,  -91.2,  11  ]],
      ...>                     [[10   ,  -30.1,  20  ],
      ...>                      [40   ,   50  , -60.1],
      ...>                      [ 0.09, -910.2, 110  ]]])[[0, 0, 0]]
      1

      iex> Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                      [ 4   ,    5  ,  -6.1],
      ...>                      [ 0.9 ,  -91.2,  11  ]],
      ...>                     [[10   ,  -30.1,  20  ],
      ...>                      [40   ,   50  , -60.1],
      ...>                      [ 0.09, -910.2, 110  ]]])[[0, 0]]
      %Tensorex{data: %{[0] => 1, [1] => -3.1, [2] => 2}, shape: [3]}

      iex> Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                      [ 4   ,    5  ,  -6.1],
      ...>                      [ 0.9 ,  -91.2,  11  ]],
      ...>                     [[10   ,  -30.1,  20  ],
      ...>                      [40   ,   50  , -60.1],
      ...>                      [ 0.09, -910.2, 110  ]]])[[0]]
      %Tensorex{data: %{[0, 0] => 1  , [0, 1] =>  -3.1, [0, 2] =>  2  ,
                        [1, 0] => 4  , [1, 1] =>   5  , [1, 2] => -6.1,
                        [2, 0] => 0.9, [2, 1] => -91.2, [2, 2] => 11  }, shape: [3, 3]}

      iex> Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                      [ 4   ,    5  ,  -6.1],
      ...>                      [ 0.9 ,  -91.2,  11  ]],
      ...>                     [[10   ,  -30.1,  20  ],
      ...>                      [40   ,   50  , -60.1],
      ...>                      [ 0.09, -910.2, 110  ]]])[[2]]
      nil

      iex> Tensorex.from_list([[ 1,  2,  3],
      ...>                     [ 4,  5,  6],
      ...>                     [ 7,  8,  9],
      ...>                     [10, 11, 12]])[[1..2]]
      %Tensorex{data: %{[0, 0] => 4, [0, 1] => 5, [0, 2] => 6,
                        [1, 0] => 7, [1, 1] => 8, [1, 2] => 9}, shape: [2, 3]}

      iex> Tensorex.from_list([[ 1,  2,  3],
      ...>                     [ 4,  5,  6],
      ...>                     [ 7,  8,  9],
      ...>                     [10, 11, 12]])[[-2..-1]]
      %Tensorex{data: %{[0, 0] =>  7, [0, 1] =>  8, [0, 2] =>  9,
                        [1, 0] => 10, [1, 1] => 11, [1, 2] => 12}, shape: [2, 3]}

      iex> Tensorex.from_list([[ 1,  2,  3],
      ...>                     [ 4,  5,  6],
      ...>                     [ 7,  8,  9],
      ...>                     [10, 11, 12]])[[1..2, 1..-1]]
      %Tensorex{data: %{[0, 0] => 5, [0, 1] => 6,
                        [1, 0] => 8, [1, 1] => 9}, shape: [2, 2]}

      iex> Tensorex.from_list([[ 1,  2,  3],
      ...>                     [ 4,  0,  6],
      ...>                     [ 7,  8,  9],
      ...>                     [10, 11, 12]])[[1, 1]]
      0.0
  """
  @spec fetch(t, [integer | Range.t(), ...]) :: {:ok, t | number} | :error
  @impl true
  def fetch(%Tensorex{data: store, shape: shape}, indices) when is_indices(indices, shape) do
    case normalize_indices(indices, shape) do
      {new_indices, false} -> {:ok, Map.get(store, new_indices, 0.0)}
      {new_indices, true} -> {:ok, slice(store, shape, new_indices)}
      :error -> :error
    end
  end

  @spec slice(
          %{optional([non_neg_integer, ...]) => number},
          [pos_integer, ...],
          [non_neg_integer | Range.t(non_neg_integer, non_neg_integer), ...]
        ) :: t
  defp slice(store, shape, indices) do
    indices_length = length(indices)
    tail_length = length(shape) - indices_length

    new_store =
      store
      |> Stream.filter(fn {index, _} -> index_in_range?(index, indices) end)
      |> Stream.map(fn {index, value} ->
        new_index =
          Stream.zip(index, indices)
          |> Stream.reject(&is_integer(elem(&1, 1)))
          |> Stream.map(fn {element, range} -> Enum.find_index(range, &(&1 === element)) end)
          |> Enum.concat(Enum.slice(index, indices_length, tail_length))

        {new_index, value}
      end)
      |> Enum.into(%{})

    new_shape =
      indices
      |> Stream.reject(&is_integer/1)
      |> Stream.map(&Enum.count/1)
      |> Enum.concat(Enum.slice(shape, indices_length, tail_length))

    %Tensorex{data: new_store, shape: new_shape}
  end

  @spec normalize_indices([integer | Range.t(), ...], [pos_integer, ...]) ::
          {[non_neg_integer | Range.t(non_neg_integer, non_neg_integer), ...],
           slice_mode :: boolean}
          | :error
  defp normalize_indices(indices, shape) do
    try do
      Stream.zip(indices, shape)
      |> Enum.map_reduce(length(indices) < length(shape), fn
        {index, dimension}, acc when is_integer(index) and index < 0 and -dimension <= index ->
          {index + dimension, acc}

        {index, dimension}, acc when is_integer(index) and 0 <= index and index < dimension ->
          {index, acc}

        {index_start..index_end, dimension}, _
        when index_start < 0 and -dimension <= index_start and
               index_end < 0 and -dimension <= index_end ->
          {(index_start + dimension)..(index_end + dimension), true}

        {index_start..index_end, dimension}, _
        when index_start < 0 and -dimension <= index_start and
               0 <= index_end and index_end < dimension ->
          {(index_start + dimension)..index_end, true}

        {index_start..index_end, dimension}, _
        when index_end < 0 and -dimension <= index_end and
               0 <= index_start and index_start < dimension ->
          {index_start..(index_end + dimension), true}

        {index_start..index_end = index, dimension}, _
        when 0 <= index_start and index_start < dimension and
               0 <= index_end and index_end < dimension ->
          {index, true}
      end)
    rescue
      FunctionClauseError -> :error
    end
  end

  @spec index_in_range?(
          [non_neg_integer, ...],
          [non_neg_integer | Range.t(non_neg_integer, non_neg_integer), ...]
        ) :: boolean
  defp index_in_range?(index, indices) do
    Stream.zip(index, indices)
    |> Enum.all?(fn
      {element, element} -> true
      {element, _.._ = range} -> element in range
      _ -> false
    end)
  end

  @doc """
  Returns a tensor or a number stored at the index and update it at the same time.

      iex> get_and_update_in(
      ...>   Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                        [ 4   ,    5  ,  -6.1],
      ...>                        [ 0.9 ,  -91.2,  11  ]],
      ...>                       [[10   ,  -30.1,  20  ],
      ...>                        [40   ,   50  , -60.1],
      ...>                        [ 0.09, -910.2, 110  ]]])[[0, 1, 0]], &{&1, &1 * 3.5})
      {4, %Tensorex{data: %{[0, 0, 0] =>  1   , [0, 0, 1] =>   -3.1, [0, 0, 2] =>   2  ,
                            [0, 1, 0] => 14.0 , [0, 1, 1] =>    5  , [0, 1, 2] =>  -6.1,
                            [0, 2, 0] =>  0.9 , [0, 2, 1] =>  -91.2, [0, 2, 2] =>  11  ,
                            [1, 0, 0] => 10   , [1, 0, 1] =>  -30.1, [1, 0, 2] =>  20  ,
                            [1, 1, 0] => 40   , [1, 1, 1] =>   50  , [1, 1, 2] => -60.1,
                            [1, 2, 0] =>  0.09, [1, 2, 1] => -910.2, [1, 2, 2] => 110  }, shape: [2, 3, 3]}}

      iex> get_and_update_in(
      ...>   Tensorex.from_list([[ 1,  2,  3],
      ...>                       [ 4,  5,  6],
      ...>                       [ 7,  8,  9],
      ...>                       [10, 11, 12]])[[1..2, 1..2]],
      ...>   &{&1, Tensorex.from_list([[13, 14],
      ...>                             [15, 16]])})
      {%Tensorex{data: %{[0, 0] => 5, [0, 1] => 6,
                         [1, 0] => 8, [1, 1] => 9}, shape: [2, 2]},
       %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2, [0, 2] =>  3,
                         [1, 0] =>  4, [1, 1] => 13, [1, 2] => 14,
                         [2, 0] =>  7, [2, 1] => 15, [2, 2] => 16,
                         [3, 0] => 10, [3, 1] => 11, [3, 2] => 12}, shape: [4, 3]}}

      iex> get_and_update_in(
      ...>   Tensorex.from_list([[ 1,  2,  3],
      ...>                       [ 4,  5,  6],
      ...>                       [ 7,  8,  9],
      ...>                       [10, 11, 12]])[[2]],
      ...>   &{&1, Tensorex.from_list([0, 0, 16])})
      {%Tensorex{data: %{[0] => 7, [1] => 8, [2] => 9}, shape: [3]},
       %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2, [0, 2] =>  3,
                         [1, 0] =>  4, [1, 1] =>  5, [1, 2] =>  6,
                                                     [2, 2] => 16,
                         [3, 0] => 10, [3, 1] => 11, [3, 2] => 12}, shape: [4, 3]}}

      iex> get_and_update_in(
      ...>   Tensorex.from_list([[ 1,  2],
      ...>                       [ 3,  4]])[[0..-1, 0..-1]],
      ...>   &{&1, Tensorex.from_list([[-2,  0],
      ...>                             [ 0, -3]])})
      {%Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2,
                         [1, 0] =>  3, [1, 1] =>  4}, shape: [2, 2]},
       %Tensorex{data: %{[0, 0] => -2,
                                       [1, 1] => -3}, shape: [2, 2]}}

      iex> get_and_update_in(
      ...>   Tensorex.zero([3, 2])[[1..-1, 1..-1]],
      ...>   &{&1, Tensorex.from_list([[ 1],
      ...>                             [-1]])})
      {%Tensorex{data: %{}, shape: [2, 1]},
       %Tensorex{data: %{[1, 1] =>  1,
                         [2, 1] => -1}, shape: [3, 2]}}
  """
  @spec get_and_update(
          t,
          [integer | Range.t(), ...],
          (t -> :pop | {any, t}) | (number -> :pop | {any, number})
        ) :: {any, t}
  @impl true
  def get_and_update(%Tensorex{data: store, shape: shape} = tensor, indices, fun)
      when is_indices(indices, shape) and is_function(fun, 1) do
    case normalize_indices(indices, shape) do
      {new_indices, true} ->
        case fun.(%{shape: partial_shape} = partial_tensor = slice(store, shape, new_indices)) do
          :pop ->
            {partial_tensor, %{tensor | data: drop(store, new_indices)}}

          {get_value, %Tensorex{shape: ^partial_shape, data: updated_store}} ->
            new_store =
              updated_store
              |> Enum.into(drop(store, new_indices), fn {index, value} ->
                {mapped_indices, remaining_partial_indices} =
                  Enum.map_reduce(new_indices, index, fn
                    element, acc when is_integer(element) -> {element, acc}
                    range, [partial_index | acc] -> {Enum.fetch!(range, partial_index), acc}
                  end)

                {mapped_indices ++ remaining_partial_indices, value}
              end)

            {get_value, %{tensor | data: new_store}}
        end

      {new_indices, false} ->
        case fun.(value = Map.get(store, new_indices, 0.0)) do
          :pop ->
            {value, %{tensor | data: Map.delete(store, new_indices)}}

          {get_value, updated_value} when updated_value == 0 ->
            {get_value, %{tensor | data: Map.delete(store, new_indices)}}

          {get_value, updated_value} when is_number(updated_value) ->
            {get_value, %{tensor | data: Map.put(store, new_indices, updated_value)}}
        end
    end
  end

  @doc """
  Pops the tensor or the number stored at the index out of the tensor.

      iex> pop_in(
      ...>   Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                        [ 4   ,    5  ,  -6.1],
      ...>                        [ 0.9 ,  -91.2,  11  ]],
      ...>                       [[10   ,  -30.1,  20  ],
      ...>                        [40   ,   50  , -60.1],
      ...>                        [ 0.09, -910.2, 110  ]]])[[0]])
      {%Tensorex{data: %{[0, 0] => 1,   [0, 1] => -3.1 , [0, 2] =>  2  ,
                         [1, 0] => 4,   [1, 1] =>  5   , [1, 2] => -6.1,
                         [2, 0] => 0.9, [2, 1] => -91.2, [2, 2] => 11  }, shape: [3, 3]},
       %Tensorex{data: %{[1, 0, 0] => 10   , [1, 0, 1] =>  -30.1, [1, 0, 2] =>  20  ,
                         [1, 1, 0] => 40   , [1, 1, 1] =>   50  , [1, 1, 2] => -60.1,
                         [1, 2, 0] =>  0.09, [1, 2, 1] => -910.2, [1, 2, 2] => 110  }, shape: [2, 3, 3]}}

      iex> pop_in(
      ...>   Tensorex.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                        [ 4   ,    5  ,  -6.1],
      ...>                        [ 0.9 ,  -91.2,  11  ]],
      ...>                       [[10   ,  -30.1,  20  ],
      ...>                        [40   ,   50  , -60.1],
      ...>                        [ 0.09, -910.2, 110  ]]])[[0, 1, 2]])
      {-6.1, %Tensorex{data: %{[0, 0, 0] =>  1   , [0, 0, 1] =>   -3.1, [0, 0, 2] =>   2  ,
                               [0, 1, 0] =>  4   , [0, 1, 1] =>    5  ,
                               [0, 2, 0] =>  0.9 , [0, 2, 1] =>  -91.2, [0, 2, 2] =>  11  ,
                               [1, 0, 0] => 10   , [1, 0, 1] =>  -30.1, [1, 0, 2] =>  20  ,
                               [1, 1, 0] => 40   , [1, 1, 1] =>   50  , [1, 1, 2] => -60.1,
                               [1, 2, 0] =>  0.09, [1, 2, 1] => -910.2, [1, 2, 2] => 110  }, shape: [2, 3, 3]}}
  """
  @spec pop(t, [integer | Range.t(), ...]) :: {t | number, t}
  @impl true
  def pop(%Tensorex{data: store, shape: shape} = tensor, indices) do
    case normalize_indices(indices, shape) do
      {new_indices, true} ->
        {slice(store, shape, new_indices), %{tensor | data: drop(store, new_indices)}}

      {new_indices, false} ->
        {Map.get(store, new_indices, 0.0), %{tensor | data: Map.delete(store, new_indices)}}
    end
  end

  @spec drop(
          %{optional([non_neg_integer, ...]) => number},
          [non_neg_integer | Range.t(non_neg_integer, non_neg_integer)]
        ) :: %{optional([non_neg_integer, ...]) => number}
  defp drop(store, indices) do
    Stream.filter(store, fn {index, _} ->
      Enum.any?(Stream.zip(index, indices), fn
        {element, element} -> false
        {element, _.._ = range} -> element not in range
        _ -> true
      end)
    end)
    |> Enum.into(%{})
  end

  defguardp is_positive_integer(number) when is_integer(number) and number > 0

  @doc """
  Returns a 2-rank tensor with all diagonal elements of 1.

      iex> Tensorex.kronecker_delta(3)
      %Tensorex{data: %{[0, 0] => 1,
                                     [1, 1] => 1,
                                                  [2, 2] => 1}, shape: [3, 3]}
  """
  @spec kronecker_delta(pos_integer) :: t
  def kronecker_delta(dimension) when is_positive_integer(dimension) do
    store = 0..(dimension - 1) |> Enum.into(%{}, &{[&1, &1], 1})
    %Tensorex{data: store, shape: [dimension, dimension]}
  end

  @doc """
  Returns a tensor with all of zero elements.

      iex> Tensorex.zero([4, 4, 2])
      %Tensorex{data: %{}, shape: [4, 4, 2]}

      iex> Tensorex.zero([-5])
      ** (ArgumentError) expected a list of positive integers, got: [-5]
  """
  @spec zero([pos_integer, ...]) :: t
  def zero(shape) when is_list(shape) and length(shape) > 0 do
    %Tensorex{data: %{}, shape: validate_shape!(shape)}
  end

  @spec validate_shape!([pos_integer, ...]) :: [pos_integer, ...]
  defp validate_shape!(shape) do
    if Enum.all?(shape, &is_positive_integer/1) do
      shape
    else
      raise ArgumentError, "expected a list of positive integers, got: #{inspect(shape)}"
    end
  end

  @doc """
  Checks if the given tensor is upper triangular or not.

      iex> Tensorex.triangular?(Tensorex.from_list([[2, 1,  3],
      ...>                                          [0, 3,  6],
      ...>                                          [0, 0, -9]]))
      true

      iex> Tensorex.triangular?(Tensorex.from_list([[2, 0,  0],
      ...>                                          [0, 3,  0],
      ...>                                          [3, 0, -9]]))
      false

      iex> Tensorex.triangular?(Tensorex.from_list([[[2,  5], [0,  1]],
      ...>                                          [[0,  0], [0, -2]],
      ...>                                          [[0,  0], [0,  0]]]))
      true

      iex> Tensorex.triangular?(Tensorex.from_list([[[2,  5], [0,  1]],
      ...>                                          [[6,  0], [0, -2]],
      ...>                                          [[0,  0], [0,  0]]]))
      false
  """
  @spec triangular?(t) :: boolean
  def triangular?(%Tensorex{shape: [_]}), do: false

  def triangular?(%Tensorex{data: store}) do
    Enum.all?(store, fn {index, _} -> Enum.sort(index) == index end)
  end

  @doc """
  Checks if the given tensor is diagonal or not.

      iex> Tensorex.diagonal?(Tensorex.from_list([[2, 0,  0],
      ...>                                        [0, 3,  0],
      ...>                                        [0, 0, -9]]))
      true

      iex> Tensorex.diagonal?(Tensorex.from_list([[ 2  , 0,  0],
      ...>                                        [ 0  , 3,  0],
      ...>                                        [-5.3, 0, -9]]))
      false
  """
  @spec diagonal?(t) :: boolean
  def diagonal?(%Tensorex{shape: [_]}), do: false

  def diagonal?(%Tensorex{data: store}) do
    Enum.all?(store, fn {index, _} -> Enum.count(Stream.uniq(index)) === 1 end)
  end

  @doc """
  Returns a tensor where each element is the result of invoking `mapper` on each corresponding
  element of the given tensor.

      iex> Tensorex.map(Tensorex.from_list([[[ 0,  1, 2], [-3, -1,  1]],
      ...>                                  [[-4, -2, 0], [ 1,  0, -1]]]), &(&1 * &1))
      %Tensorex{data: %{                 [0, 0, 1] => 1, [0, 0, 2] => 4, [0, 1, 0] => 9, [0, 1, 1] => 1, [0, 1, 2] => 1,
                        [1, 0, 0] => 16, [1, 0, 1] => 4,                 [1, 1, 0] => 1,                 [1, 1, 2] => 1}, shape: [2, 2, 3]}

      iex> Tensorex.map(Tensorex.from_list([[[ 0,  1, 2], [-3, -1,  1]],
      ...>                                  [[-4, -2, 0], [ 1,  0, -1]]]), &(&1 + 3))
      %Tensorex{data: %{[0, 0, 0] =>  3.0, [0, 0, 1] => 4, [0, 0, 2] => 5  ,                 [0, 1, 1] => 2  , [0, 1, 2] => 4,
                        [1, 0, 0] => -1  , [1, 0, 1] => 1, [1, 0, 2] => 3.0, [1, 1, 0] => 4, [1, 1, 1] => 3.0, [1, 1, 2] => 2}, shape: [2, 2, 3]}

      iex> Tensorex.map(Tensorex.from_list([[-3, -1,  1],
      ...>                                  [-4, -2,  0],
      ...>                                  [ 1,  0, -1]]),
      ...>              fn
      ...>                value, [index, index] -> value * value
      ...>                value, _ -> value
      ...>              end)
      %Tensorex{data: %{[0, 0] =>  9, [0, 1] => -1, [0, 2] => 1,
                        [1, 0] => -4, [1, 1] =>  4,
                        [2, 0] =>  1,               [2, 2] => 1}, shape: [3, 3]}
  """
  @spec map(t, ([pos_integer, ...], number -> number) | (number -> number)) :: t
  def map(%Tensorex{data: store, shape: shape} = tensor, mapper) when is_function(mapper, 2) do
    mapped_store =
      all_indices(shape)
      |> Stream.flat_map(fn index ->
        case mapper.(Map.get(store, index, 0.0), index) do
          value when value == 0 -> []
          value -> [{index, value}]
        end
      end)
      |> Enum.into(%{})

    %{tensor | data: mapped_store}
  end

  def map(%Tensorex{} = tensor, mapper) when is_function(mapper, 1) do
    map(tensor, fn value, _ -> mapper.(value) end)
  end

  @spec all_indices([pos_integer, ...]) :: Enum.t()
  defp all_indices([dimension]), do: Stream.map(0..(dimension - 1), &[&1])

  defp all_indices([dimension | shape]) do
    all_indices(shape)
    |> Stream.map(fn indices -> Stream.map(0..(dimension - 1), &[&1 | indices]) end)
    |> Stream.concat()
  end

  @doc """
  Returns a tensor where all of elements are the given value.

      iex> Tensorex.fill([3, 4, 2], 2)
      %Tensorex{data: %{[0, 0, 0] => 2, [0, 0, 1] => 2,
                        [0, 1, 0] => 2, [0, 1, 1] => 2,
                        [0, 2, 0] => 2, [0, 2, 1] => 2,
                        [0, 3, 0] => 2, [0, 3, 1] => 2,
                        [1, 0, 0] => 2, [1, 0, 1] => 2,
                        [1, 1, 0] => 2, [1, 1, 1] => 2,
                        [1, 2, 0] => 2, [1, 2, 1] => 2,
                        [1, 3, 0] => 2, [1, 3, 1] => 2,
                        [2, 0, 0] => 2, [2, 0, 1] => 2,
                        [2, 1, 0] => 2, [2, 1, 1] => 2,
                        [2, 2, 0] => 2, [2, 2, 1] => 2,
                        [2, 3, 0] => 2, [2, 3, 1] => 2}, shape: [3, 4, 2]}

      iex> Tensorex.fill([2, 2, 5], 0.0)
      %Tensorex{data: %{}, shape: [2, 2, 5]}
  """
  @spec fill([pos_integer, ...], number) :: t
  def fill(shape, value) when is_list(shape) and length(shape) > 0 and value == 0, do: zero(shape)

  def fill(shape, value) when is_list(shape) and length(shape) > 0 and is_number(value) do
    store = shape |> validate_shape!() |> all_indices() |> Enum.into(%{}, &{&1, value})
    %Tensorex{data: store, shape: shape}
  end

  @doc """
  Updates dimensions of each order.

  If new shape has larger dimension than previous one, values at the increased indices are
  considered to be zero. Otherwise if new shape has less dimension, it discards values at the
  removed indices.

      iex> Tensorex.reshape(Tensorex.from_list([[[ 1,  2,  3],
      ...>                                       [ 4,  5,  6],
      ...>                                       [ 7,  8,  9]],
      ...>                                      [[-1, -2, -3],
      ...>                                       [-4, -5, -6],
      ...>                                       [-7, -8, -9]]]), [2, 2, 3])
      %Tensorex{data: %{[0, 0, 0] =>  1, [0, 0, 1] =>  2, [0, 0, 2] =>  3,
                        [0, 1, 0] =>  4, [0, 1, 1] =>  5, [0, 1, 2] =>  6,
                        [1, 0, 0] => -1, [1, 0, 1] => -2, [1, 0, 2] => -3,
                        [1, 1, 0] => -4, [1, 1, 1] => -5, [1, 1, 2] => -6}, shape: [2, 2, 3]}

      iex> Tensorex.reshape(Tensorex.from_list([[[ 1,  2,  3],
      ...>                                       [ 4,  5,  6],
      ...>                                       [ 7,  8,  9]],
      ...>                                      [[-1, -2, -3],
      ...>                                       [-4, -5, -6],
      ...>                                       [-7, -8, -9]]]), [3, 3, 3])
      %Tensorex{data: %{[0, 0, 0] =>  1, [0, 0, 1] =>  2, [0, 0, 2] =>  3,
                        [0, 1, 0] =>  4, [0, 1, 1] =>  5, [0, 1, 2] =>  6,
                        [0, 2, 0] =>  7, [0, 2, 1] =>  8, [0, 2, 2] =>  9,
                        [1, 0, 0] => -1, [1, 0, 1] => -2, [1, 0, 2] => -3,
                        [1, 1, 0] => -4, [1, 1, 1] => -5, [1, 1, 2] => -6,
                        [1, 2, 0] => -7, [1, 2, 1] => -8, [1, 2, 2] => -9}, shape: [3, 3, 3]}

      iex> Tensorex.reshape(Tensorex.from_list([[[ 1,  2,  3],
      ...>                                       [ 4,  5,  6],
      ...>                                       [ 7,  8,  9]],
      ...>                                      [[-1, -2, -3],
      ...>                                       [-4, -5, -6],
      ...>                                       [-7, -8, -9]]]), [3, 2, 2])
      %Tensorex{data: %{[0, 0, 0] =>  1, [0, 0, 1] =>  2,
                        [0, 1, 0] =>  4, [0, 1, 1] =>  5,
                        [1, 0, 0] => -1, [1, 0, 1] => -2,
                        [1, 1, 0] => -4, [1, 1, 1] => -5}, shape: [3, 2, 2]}
  """
  @spec reshape(t, [pos_integer, ...]) :: t
  def reshape(%Tensorex{data: store, shape: prev_shape}, shape)
      when is_list(shape) and length(prev_shape) === length(shape) do
    new_store =
      store
      |> Stream.filter(fn {indices, _} ->
        Stream.zip(indices, shape) |> Enum.all?(&(elem(&1, 0) < elem(&1, 1)))
      end)
      |> Enum.into(%{})

    %Tensorex{data: new_store, shape: shape}
  end

  @doc """
  Returns if all corresponding elements are in (relative) tolerance or not.

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[0.0000001,  0.9999998],
      ...>                                            [2.0      , -0.0000003]]),
      ...>                        Tensorex.from_list([[0        ,  1        ],
      ...>                                            [2        ,  0        ]]), 1.0e-6)
      true

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[0,  1],
      ...>                                            [2, -1],
      ...>                                            [3,  2]]),
      ...>                        Tensorex.from_list([[0,  1],
      ...>                                            [2, -1]]), 1.0e-8)
      false

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[0        , 1],
      ...>                                            [2.0      , 0]]),
      ...>                        Tensorex.from_list([[0        , 1],
      ...>                                            [2.000003 , 0]]), 1.0e-6)
      false

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[1.8200340109e62, 1.0e-52      ],
      ...>                                            [2.335142153e-41, 0            ]]),
      ...>                        Tensorex.from_list([[1.8200338243e62, 1.0000009e-52],
      ...>                                            [2.335142296e-41, 3.242e-7     ]]), 1.0e-6)
      true

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[1.8200440109e62, 1.0e-52     ],
      ...>                                            [2.335142296e-41, 0           ]]),
      ...>                        Tensorex.from_list([[1.8200440109e62, 1.000002e-52],
      ...>                                            [2.335142296e-41, 0           ]]), 1.0e-6)
      false

      iex> Tensorex.in_tolerance?(Tensorex.from_list([[1,  3],
      ...>                                            [2,  0]]),
      ...>                        Tensorex.from_list([[1,  3],
      ...>                                            [2, -5]]), 1.0e-6)
      false
  """
  @spec in_tolerance?(t, t, number) :: boolean
  def in_tolerance?(
        %Tensorex{data: store1, shape: shape},
        %Tensorex{data: store2, shape: shape},
        tolerance
      )
      when is_number(tolerance) and tolerance >= 0 do
    Map.merge(store1, store2, fn _, value1, value2 ->
      (value1 - value2) / max(abs(value1), abs(value2))
    end)
    |> Enum.all?(&(abs(elem(&1, 1)) <= tolerance))
  end

  def in_tolerance?(%Tensorex{}, %Tensorex{}, tolerance)
      when is_number(tolerance) and tolerance >= 0 do
    false
  end

  @doc """
  Returns a permutation tensor (also called Levi-Civita or Eddington tensor).

      iex> Tensorex.permutation(3)
      %Tensorex{data: %{[0, 1, 2] =>  1, [0, 2, 1] => -1,
                        [1, 0, 2] => -1, [1, 2, 0] =>  1,
                        [2, 0, 1] =>  1, [2, 1, 0] => -1}, shape: [3, 3, 3]}
  """
  @spec permutation(pos_integer) :: t
  def permutation(dimension) when is_integer(dimension) and dimension >= 2 do
    store =
      Stream.iterate([{[], 0}], fn acc ->
        Stream.map(acc, fn {index, inversions} ->
          Stream.iterate(0, &(&1 + 1))
          |> Stream.take(dimension)
          |> Stream.reject(&(&1 in index))
          |> Stream.map(fn i -> {[i | index], Enum.count(index, &(&1 < i)) + inversions} end)
        end)
        |> Stream.concat()
      end)
      |> Enum.at(dimension)
      |> Enum.into(%{}, fn
        {index, inversions} when rem(inversions, 2) > 0 -> {index, -1}
        {index, _} -> {index, 1}
      end)

    %Tensorex{data: store, shape: List.duplicate(dimension, dimension)}
  end

  @doc """
  Inserts elements into the given index.

      iex> Tensorex.insert_at(
      ...>   Tensorex.from_list([[ 1,  2,  3],
      ...>                       [ 4,  5,  6],
      ...>                       [ 7,  8,  9],
      ...>                       [10, 11, 12]]),
      ...>   1,
      ...>   0,
      ...>   Tensorex.from_list([13, 14, 15])
      ...> )
      %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2, [0, 2] =>  3,
                        [1, 0] => 13, [1, 1] => 14, [1, 2] => 15,
                        [2, 0] =>  4, [2, 1] =>  5, [2, 2] =>  6,
                        [3, 0] =>  7, [3, 1] =>  8, [3, 2] =>  9,
                        [4, 0] => 10, [4, 1] => 11, [4, 2] => 12}, shape: [5, 3]}

      iex> Tensorex.insert_at(
      ...>   Tensorex.from_list([[ 1,  2,  3],
      ...>                       [ 4,  5,  6],
      ...>                       [ 7,  8,  9],
      ...>                       [10, 11, 12]]),
      ...>   2,
      ...>   1,
      ...>   Tensorex.from_list([13, 14, 15, 16])
      ...> )
      %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2, [0, 2] => 13, [0, 3] =>  3,
                        [1, 0] =>  4, [1, 1] =>  5, [1, 2] => 14, [1, 3] =>  6,
                        [2, 0] =>  7, [2, 1] =>  8, [2, 2] => 15, [2, 3] =>  9,
                        [3, 0] => 10, [3, 1] => 11, [3, 2] => 16, [3, 3] => 12}, shape: [4, 4]}

      iex> Tensorex.insert_at(
      ...>   Tensorex.from_list([10, 11, 12]),
      ...>   1,
      ...>   0,
      ...>   13
      ...> )
      %Tensorex{data: %{[0] => 10, [1] => 13, [2] => 11, [3] => 12}, shape: [4]}

      iex> Tensorex.insert_at(
      ...>   Tensorex.from_list([10, 11, 12]),
      ...>   1,
      ...>   0,
      ...>   0
      ...> )
      %Tensorex{data: %{[0] => 10, [2] => 11, [3] => 12}, shape: [4]}

      iex> Tensorex.insert_at(
      ...>   Tensorex.from_list([[ 1,  2,  3],
      ...>                       [ 4,  5,  6],
      ...>                       [ 7,  8,  9],
      ...>                       [10, 11, 12]]),
      ...>   2,
      ...>   1,
      ...>   Tensorex.from_list([13, 14, 15, 16, 17])
      ...> )
      ** (RuntimeError) expected the shape of the inserted tensor to be [4], got: [5]
  """
  @spec insert_at(t, non_neg_integer, non_neg_integer, t | number) :: t
  def insert_at(%Tensorex{data: store, shape: [dimension]}, index, 0, value)
      when is_integer(index) and index >= 0 and index < dimension and is_number(value) do
    keys = Map.keys(store) |> Enum.filter(fn [i] -> i >= index end)
    {tail, head} = Map.split(store, keys)
    shifted = Enum.into(tail, head, fn {[i], v} -> {[i + 1], v} end)
    new_store = if value == 0, do: shifted, else: Map.put(shifted, [index], value)
    %Tensorex{data: new_store, shape: [dimension + 1]}
  end

  def insert_at(
        %Tensorex{data: store, shape: shape},
        index,
        axis,
        %Tensorex{data: sub_store, shape: sub_shape}
      )
      when is_integer(index) and index >= 0 and
             is_integer(axis) and axis >= 0 and
             length(shape) > 1 and axis < length(shape) do
    unless (expected_shape = List.delete_at(shape, axis)) == sub_shape do
      raise "expected the shape of the inserted tensor to be " <>
              "#{inspect(expected_shape)}, got: #{inspect(sub_shape)}"
    end

    keys = Map.keys(store) |> Enum.filter(&(Enum.at(&1, axis) >= index))
    {tail, head} = Map.split(store, keys)

    new_store =
      Enum.into(tail, head, fn {i, v} -> {List.update_at(i, axis, &(&1 + 1)), v} end)
      |> Map.merge(
        Enum.into(sub_store, %{}, fn {i, v} -> {List.insert_at(i, axis, index), v} end)
      )

    {dimension, ^sub_shape} = List.pop_at(shape, axis)
    %Tensorex{data: new_store, shape: List.insert_at(sub_shape, axis, dimension + 1)}
  end

  @doc """
  Removes elements at given index and closes up indices.

      iex> Tensorex.delete_at(Tensorex.from_list([[ 1,  2,  3],
      ...>                                        [ 4,  5,  6],
      ...>                                        [ 7,  8,  9],
      ...>                                        [10, 11, 12]]), 2, 0)
      %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  2, [0, 2] =>  3,
                        [1, 0] =>  4, [1, 1] =>  5, [1, 2] =>  6,
                        [2, 0] => 10, [2, 1] => 11, [2, 2] => 12}, shape: [3, 3]}

      iex> Tensorex.delete_at(Tensorex.from_list([[ 1,  2,  3],
      ...>                                        [ 4,  5,  6],
      ...>                                        [ 7,  8,  9],
      ...>                                        [10, 11, 12]]), 1, 1)
      %Tensorex{data: %{[0, 0] =>  1, [0, 1] =>  3,
                        [1, 0] =>  4, [1, 1] =>  6,
                        [2, 0] =>  7, [2, 1] =>  9,
                        [3, 0] => 10, [3, 1] => 12}, shape: [4, 2]}

      iex> Tensorex.delete_at(Tensorex.from_list([1, 2, 3, 4, 5, 6]), 2, 0)
      %Tensorex{data: %{[0] =>  1, [1] =>  2, [2] =>  4, [3] =>  5, [4] =>  6}, shape: [5]}
  """
  @spec delete_at(t, non_neg_integer, non_neg_integer) :: t
  def delete_at(%Tensorex{data: store, shape: shape}, index, axis)
      when is_integer(index) and index >= 0 and
             is_integer(axis) and axis >= 0 and axis < length(shape) do
    %{shift: shift_keys, drop: drop_keys} =
      Map.keys(store)
      |> Enum.group_by(fn key ->
        case Enum.at(key, axis) do
          ^index -> :drop
          i when i < index -> :leave
          _ -> :shift
        end
      end)

    {tail, head} = Map.drop(store, drop_keys) |> Map.split(shift_keys)
    new_store = Enum.into(tail, head, fn {i, v} -> {List.update_at(i, axis, &(&1 - 1)), v} end)
    %Tensorex{data: new_store, shape: List.update_at(shape, axis, &(&1 - 1))}
  end
end
