defmodule Tensorex do
  @moduledoc """
  The module to oparate with tensors.
  """
  @typep data :: %{pos_integer => data | number}
  @opaque t :: %__MODULE__{data: data | nil, shape: [pos_integer]}
  defstruct [:data, :shape]
  @type data_list :: [data_list | number]
  @doc """
  Create a new tensor from list.

      iex> #{__MODULE__}.from_list([1.1, 2.1, -5.3, 4])
      %#{__MODULE__}{data: %{1 => 1.1, 2 => 2.1, 3 => -5.3, 4 => 4}, shape: [4]}

      iex> #{__MODULE__}.from_list([[1.1,  2.1, -5.3, 4  ],
      ...>                          [0.8, -8,   21.4, 3.3]])
      %#{__MODULE__}{data: %{1 => %{1 => 1.1, 2 =>  2.1, 3 => -5.3, 4 => 4  },
                             2 => %{1 => 0.8, 2 => -8,   3 => 21.4, 4 => 3.3}}, shape: [2, 4]}

      iex> #{__MODULE__}.from_list([[[0.0, 0.0, 0.0],
      ...>                           [0.0, 0.0, 0.0]],
      ...>                          [[0.0, 0.0, 0.0],
      ...>                           [0.0, 0.0, 0.0]]])
      %#{__MODULE__}{data: nil, shape: [2, 2, 3]}
  """
  @spec from_list(data_list) :: t
  def from_list(d) when is_list(d) do
    cond do
      Enum.all?(d, &(&1 == 0)) ->
        %__MODULE__{data: nil, shape: [length(d)]}

      Enum.all?(d, &is_number/1) ->
        %__MODULE__{
          data:
            d
            |> Stream.with_index(1)
            |> Stream.filter(&(elem(&1, 0) != 0))
            |> Enum.into(%{}, &{elem(&1, 1), elem(&1, 0)}),
          shape: [length(d)]
        }

      true ->
        d = Enum.map(d, &from_list/1)
        [s] = Stream.map(d, & &1.shape) |> Enum.uniq()

        if Enum.all?(d, &is_nil(&1.data)) do
          %__MODULE__{data: nil, shape: [length(d) | s]}
        else
          %__MODULE__{
            data:
              d
              |> Stream.map(& &1.data)
              |> Stream.with_index(1)
              |> Stream.filter(&elem(&1, 0))
              |> Enum.into(%{}, &{elem(&1, 1), elem(&1, 0)}),
            shape: [length(d) | s]
          }
        end
    end
  end

  @behaviour Access
  @doc """
  Returns a tensor or a number stored at the index.

      iex> #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                           [ 4   ,    5  ,  -6.1],
      ...>                           [ 0.9 ,  -91.2,  11  ]],
      ...>                          [[10   ,  -30.1,  20  ],
      ...>                           [40   ,   50  , -60.1],
      ...>                           [ 0.09, -910.2, 110  ]]])[1][1][1]
      1

      iex> #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                           [ 4   ,    5  ,  -6.1],
      ...>                           [ 0.9 ,  -91.2,  11  ]],
      ...>                          [[10   ,  -30.1,  20  ],
      ...>                           [40   ,   50  , -60.1],
      ...>                           [ 0.09, -910.2, 110  ]]])[1][1]
      %#{__MODULE__}{data: %{1 => 1, 2 => -3.1, 3 => 2}, shape: [3]}

      iex> #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                           [ 4   ,    5  ,  -6.1],
      ...>                           [ 0.9 ,  -91.2,  11  ]],
      ...>                          [[10   ,  -30.1,  20  ],
      ...>                           [40   ,   50  , -60.1],
      ...>                           [ 0.09, -910.2, 110  ]]])[1]
      %#{__MODULE__}{data: %{1 => %{1 => 1  , 2 =>  -3.1, 3 =>  2  },
                             2 => %{1 => 4  , 2 =>   5  , 3 => -6.1},
                             3 => %{1 => 0.9, 2 => -91.2, 3 => 11  }}, shape: [3, 3]}

      iex> #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                           [ 4   ,    5  ,  -6.1],
      ...>                           [ 0.9 ,  -91.2,  11  ]],
      ...>                          [[10   ,  -30.1,  20  ],
      ...>                           [40   ,   50  , -60.1],
      ...>                           [ 0.09, -910.2, 110  ]]])[3]
      nil
  """
  @impl true
  def fetch(%__MODULE__{data: nil, shape: [d]}, i) when i in 1..d, do: {:ok, 0.0}

  def fetch(%__MODULE__{data: v, shape: [d]}, i) when i in 1..d do
    {:ok, Map.get(v, i, 0.0)}
  end

  def fetch(%__MODULE__{data: nil, shape: [d | s]}, i) when i in 1..d do
    {:ok, %__MODULE__{shape: s}}
  end

  def fetch(%__MODULE__{data: v, shape: [d | s]}, i) when i in 1..d do
    {:ok, %__MODULE__{data: Map.get(v, i), shape: s}}
  end

  def fetch(_, _), do: :error

  @doc """
  Returns a tensor or a number stored at the index and update it at the same time.

      iex> get_and_update_in(
      ...>   #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                             [ 4   ,    5  ,  -6.1],
      ...>                             [ 0.9 ,  -91.2,  11  ]],
      ...>                            [[10   ,  -30.1,  20  ],
      ...>                             [40   ,   50  , -60.1],
      ...>                             [ 0.09, -910.2, 110  ]]])[1][2][1], &{&1, &1 * 3.5})
      {4, %#{__MODULE__}{data: %{1 => %{1 => %{1 =>  1   , 2 =>   -3.1, 3 =>   2  },
                                        2 => %{1 => 14.0 , 2 =>    5  , 3 =>  -6.1},
                                        3 => %{1 =>  0.9 , 2 =>  -91.2, 3 =>  11  }},
                                 2 => %{1 => %{1 => 10   , 2 =>  -30.1, 3 =>  20  },
                                        2 => %{1 => 40   , 2 =>   50  , 3 => -60.1},
                                        3 => %{1 =>  0.09, 2 => -910.2, 3 => 110  }}}, shape: [2, 3, 3]}}
  """
  @impl true
  def get_and_update(t, i, f) do
    case fetch(t, i) do
      {:ok, v} ->
        case f.(v) do
          :pop ->
            pop(t, i)

          {p, u} ->
            {p, update(t, i, u)}

          other ->
            raise "the given function must return a two-element tuple or :pop, got: " <>
                    inspect(other)
        end

      :error ->
        raise "expected the index to be between 1 and #{List.first(t.shape)}, got: #{i}"
    end
  end

  @spec update(t, pos_integer, t | number) :: t
  defp update(%__MODULE__{data: v, shape: [_]} = t, i, u)
       when is_map_key(v, i) and map_size(v) > 1 and u == 0 do
    %{t | data: Map.delete(v, i)}
  end

  defp update(%__MODULE__{data: v, shape: [_]} = t, i, u) when is_map_key(v, i) and u == 0 do
    %{t | data: nil}
  end

  defp update(%__MODULE__{shape: [_]} = t, _, u) when u == 0, do: t

  defp update(%__MODULE__{data: nil, shape: [_]} = t, i, u) when is_number(u) do
    %{t | data: %{i => u}}
  end

  defp update(%__MODULE__{data: v, shape: [_]} = t, i, u) when is_number(u) do
    %{t | data: Map.put(v, i, u)}
  end

  defp update(%__MODULE__{shape: [_]}, _, u) do
    raise "expected the updated value to be a number, got: #{inspect(u)}"
  end

  defp update(%__MODULE__{data: v, shape: [_ | s]} = t, i, %__MODULE__{data: nil, shape: s})
       when map_size(v) > 1 and is_map_key(v, i) do
    %{t | data: nil}
  end

  defp update(%__MODULE__{data: v, shape: [_ | s]} = t, i, %__MODULE__{data: nil, shape: s})
       when is_map_key(v, i) do
    %{t | data: Map.delete(v, i)}
  end

  defp update(%__MODULE__{shape: [_ | s]} = t, _, %__MODULE__{data: nil, shape: s}), do: t

  defp update(%__MODULE__{data: nil, shape: [_ | s]} = t, i, %__MODULE__{data: u, shape: s}) do
    %{t | data: %{i => u}}
  end

  defp update(%__MODULE__{data: v, shape: [_ | s]} = t, i, %__MODULE__{data: u, shape: s}) do
    %{t | data: Map.put(v, i, u)}
  end

  defp update(%__MODULE__{shape: s}, _, u) do
    raise "expected the updated value to be a #{Enum.join(s, "×")} tensor, got: #{inspect(u)}"
  end

  @doc """
  Pop the tensor or the number stored at the index out of the tensor.

      iex> Access.pop(
      ...>   #{__MODULE__}.from_list([[[ 1   ,   -3.1,   2  ],
      ...>                             [ 4   ,    5  ,  -6.1],
      ...>                             [ 0.9 ,  -91.2,  11  ]],
      ...>                            [[10   ,  -30.1,  20  ],
      ...>                             [40   ,   50  , -60.1],
      ...>                             [ 0.09, -910.2, 110  ]]]), 1)
      {
        %#{__MODULE__}{data: %{1 => %{1 => 1, 2 => -3.1, 3 => 2},
                               2 => %{1 => 4, 2 => 5, 3 => -6.1},
                               3 => %{1 => 0.9, 2 => -91.2, 3 => 11}}, shape: [3, 3]},
        %#{__MODULE__}{data: %{2 => %{1 => %{1 => 10   , 2 =>  -30.1, 3 =>  20  },
                                      2 => %{1 => 40   , 2 =>   50  , 3 => -60.1},
                                      3 => %{1 =>  0.09, 2 => -910.2, 3 => 110  }}}, shape: [2, 3, 3]}}
  """
  @impl true
  def pop(%__MODULE__{data: v, shape: [_]} = t, i) when is_map_key(v, i) do
    {v[i], %{t | data: Map.delete(v, i)}}
  end

  def pop(%__MODULE__{shape: [d]} = t, i) when i in 1..d, do: {0.0, t}

  def pop(%__MODULE__{data: v} = t, i) when is_map_key(v, i) and map_size(v) > 1 do
    {t[i], %{t | data: Map.delete(v, i)}}
  end

  def pop(%__MODULE__{data: v} = t, i) when is_map_key(v, i), do: {t[i], %{t | data: nil}}
  def pop(%__MODULE__{shape: [d | _]} = t, i) when i in 1..d, do: {t[i], t}

  @doc """
  Adds tensors.

      iex> #{__MODULE__}.add(
      ...>   #{__MODULE__}.from_list([[0,  1  ,  2  ],
      ...>                            [3, -4  , -5.5]]),
      ...>   #{__MODULE__}.from_list([[3, -2  , -2  ],
      ...>                            [6, -8.1, 12  ]]))
      %#{__MODULE__}{data: %{1 => %{1 => 3, 2 =>  -1            },
                             2 => %{1 => 9, 2 => -12.1, 3 => 6.5}}, shape: [2, 3]}

      iex> #{__MODULE__}.add(
      ...>   #{__MODULE__}.from_list([[0  ,  1  ,  2  ],
      ...>                            [3  , -4  , -5.5]]),
      ...>   #{__MODULE__}.from_list([[0.0, -1  , -2  ],
      ...>                            [6  , -8.1, 12  ]]))
      %#{__MODULE__}{data: %{2 => %{1 => 9, 2 => -12.1, 3 => 6.5}}, shape: [2, 3]}
  """
  @spec add(t, t) :: t
  def add(%__MODULE__{data: nil, shape: s}, %__MODULE__{shape: s} = t), do: t
  def add(%__MODULE__{shape: s} = t, %__MODULE__{data: nil, shape: s}), do: t

  def add(%__MODULE__{data: d1, shape: s} = t, %__MODULE__{data: d2, shape: s}) do
    %{t | data: merge(d1, d2, :add)}
  end

  @spec merge(data, data, any) :: data | nil
  defp merge(%{} = d1, %{} = d2, f) do
    Enum.reduce(d2, d1, &merge(&1, &2, f)) |> empty_map_to_nil()
  end

  defp merge({i, v}, d, f) when is_map_key(d, i) do
    case merge(d[i], v, f) do
      nil -> Map.delete(d, i)
      u -> Map.put(d, i, u)
    end
  end

  defp merge({i, v}, d, :add), do: Map.put(d, i, v)
  defp merge({i, v}, d, :sub), do: Map.put(d, i, operate(v, :negate))
  defp merge(d1, d2, :add) when d1 + d2 == 0, do: nil
  defp merge(d1, d2, :add), do: d1 + d2
  defp merge(d1, d2, :sub) when d1 == d2, do: nil
  defp merge(d1, d2, :sub), do: d1 - d2
  @spec empty_map_to_nil(map) :: map | nil
  defp empty_map_to_nil(%{} = m) when map_size(m) === 0, do: nil
  defp empty_map_to_nil(m), do: m

  @doc """
  Subtracts a tensor from another.

      iex> #{__MODULE__}.subtract(
      ...>   #{__MODULE__}.from_list([[0,  1,  2], [3, -4,   -5.5]]),
      ...>   #{__MODULE__}.from_list([[3, -2, -2], [6, -8.1, 12  ]]))
      %#{__MODULE__}{data: %{1 => %{1 => -3, 2 => 3  , 3 =>   4  },
                             2 => %{1 => -3, 2 => 4.1, 3 => -17.5}}, shape: [2, 3]}

      iex> #{__MODULE__}.subtract(
      ...>   #{__MODULE__}.from_list([[0,   1, 2], [3, -4,   -5.5]]),
      ...>   #{__MODULE__}.from_list([[0.0, 1, 2], [6, -8.1, 12  ]]))
      %#{__MODULE__}{data: %{2 => %{1 => -3, 2 => 4.1, 3 => -17.5}}, shape: [2, 3]}
  """
  @spec subtract(t, t) :: t
  def subtract(%__MODULE__{data: nil, shape: s}, %__MODULE__{shape: s} = t), do: negate(t)
  def subtract(%__MODULE__{shape: s} = t, %__MODULE__{data: nil, shape: s}), do: t

  def subtract(%__MODULE__{data: d1, shape: s} = t, %__MODULE__{data: d2, shape: s}) do
    %{t | data: merge(d1, d2, :sub)}
  end

  @doc """
  Negates a tensor.

      iex> #{__MODULE__}.negate(
      ...>   #{__MODULE__}.from_list([[ 2  , 3.5, -4  , 0  ],
      ...>                            [-2.2, 6  ,  0.0, 5.5]]))
      %#{__MODULE__}{data: %{1 => %{1 => -2  , 2 => -3.5, 3 => 4           },
                             2 => %{1 =>  2.2, 2 => -6  ,         4 => -5.5}}, shape: [2, 4]}
  """
  @spec negate(t) :: t
  def negate(%__MODULE__{data: nil} = t), do: t
  def negate(%__MODULE__{data: d} = t), do: %{t | data: operate(d, :negate)}
  @spec operate(data, any) :: data
  defp operate(%{} = d, f), do: Enum.into(d, %{}, &{elem(&1, 0), operate(elem(&1, 1), f)})
  defp operate(d, :negate), do: -d

  @doc """
  Makes a product of tensors.

      iex> #{__MODULE__}.multiply(
      ...>   #{__MODULE__}.from_list([2, 5.2, -4  , 0  ]),
      ...>   #{__MODULE__}.from_list([2, 3.5, -1.6, 8.2]))
      %#{__MODULE__}{data: %{1 => %{1 => 4   , 2 =>   7.0, 3 => -3.2 , 4 => 16.4 },
                             2 => %{1 => 10.4, 2 =>  18.2, 3 => -8.32, 4 => 42.64},
                             3 => %{1 => -8  , 2 => -14.0, 3 =>  6.4 , 4 => -32.8}}, shape: [4, 4]}

      iex> #{__MODULE__}.multiply(3.5,
      ...>   #{__MODULE__}.from_list([[2   ,  3.5, -1.5, 8.0],
      ...>                            [4.12, -2  ,  1  , 0  ]]))
      %#{__MODULE__}{data: %{1 => %{1 =>  7.0 , 2 => 12.25, 3 => -5.25, 4 => 28.0},
                             2 => %{1 => 14.42, 2 => -7.0 , 3 =>  3.5            }}, shape: [2, 4]}
  """
  @spec multiply(t | number, t | number) :: t
  def multiply(%__MODULE__{data: nil, shape: s1} = t, %__MODULE__{shape: s2}) do
    %{t | shape: s1 ++ s2}
  end

  def multiply(%__MODULE__{shape: s1}, %__MODULE__{data: nil, shape: s2} = t) do
    %{t | shape: s1 ++ s2}
  end

  def multiply(%__MODULE__{data: d1, shape: s1}, %__MODULE__{data: d2, shape: s2}) do
    %__MODULE__{data: replace(d1, d2, :prod), shape: s1 ++ s2}
  end

  def multiply(%__MODULE__{} = t, s) when is_number(s), do: multiply(s, t)
  def multiply(s, %__MODULE__{data: nil} = t) when is_number(s), do: t

  def multiply(s, %__MODULE__{data: d} = t) when is_number(s) do
    %{t | data: replace(s, d, :prod)}
  end

  @doc """
  Makes a dot product of tensors.

  Components specified by the `axes` argument will be sumed up.

      iex> #{__MODULE__}.multiply(
      ...>   #{__MODULE__}.from_list([0, 0.0,  0.0, 0  ]),
      ...>   #{__MODULE__}.from_list([2, 3.5, -1.6, 8.2]), [{0, 0}])
      0.0

      iex> #{__MODULE__}.multiply(
      ...>   #{__MODULE__}.from_list([[2  , 3.5, -1.6,   8.2],
      ...>                            [1.1, 3.0,  8  , -12.1]]),
      ...>   #{__MODULE__}.from_list([[0  , 0.0],
      ...>                            [0.0, 0  ],
      ...>                            [0.0, 0  ],
      ...>                            [0  , 0  ]]), [{0, 1}, {1, 0}])
      0.0

      iex> #{__MODULE__}.multiply(
      ...>   #{__MODULE__}.from_list([2, 5.2, -4  , 0  ]),
      ...>   #{__MODULE__}.from_list([2, 3.5, -1.6, 8.2]), [{0, 0}])
      28.6

      iex> #{__MODULE__}.multiply(
      ...>   #{__MODULE__}.from_list([[ 2   ,  5.5, -4  , 0  ],
      ...>                            [ 4.12, -2  ,  1  , 0  ]]),
      ...>   #{__MODULE__}.from_list([[ 2   ,  3.5],
      ...>                            [-1.6 ,  8.2],
      ...>                            [ 2   , -3.5],
      ...>                            [-1.5 ,  8.0]]), [{0, 1}])
      %#{__MODULE__}{data: %{1 => %{1 => 18.42, 2 =>  30.584, 3 => -10.42, 4 =>  29.96},
                             2 => %{1 =>  4.0 , 2 => -25.2  , 3 =>  18.0 , 4 => -24.25},
                             3 => %{1 => -4.5 , 2 =>  14.6  , 3 => -11.5 , 4 =>  14.0 }}, shape: [4, 4]}
  """
  @spec multiply(t, t, axes :: [{non_neg_integer, non_neg_integer}]) :: t | number
  def multiply(%__MODULE__{data: nil, shape: s1} = t, %__MODULE__{shape: s2}, a) do
    {a1, a2} = Enum.unzip(a)
    s = Enum.flat_map([{s1, a1}, {s2, a2}], &contract_shape/1)
    if s == [], do: 0.0, else: %{t | shape: s}
  end

  def multiply(%__MODULE__{shape: s1}, %__MODULE__{data: nil, shape: s2} = t, a) do
    {a1, a2} = Enum.unzip(a)
    s = Enum.flat_map([{s1, a1}, {s2, a2}], &contract_shape/1)
    if s == [], do: 0.0, else: %{t | shape: s}
  end

  def multiply(%__MODULE__{data: d1, shape: s1}, %__MODULE__{data: d2, shape: s2}, a) do
    {a1, a2} = Enum.unzip(a)
    d = combination(swap_axes(d1, a1), swap_axes(d2, a2), Enum.count(a))
    s = Enum.flat_map([{s1, a1}, {s2, a2}], &contract_shape/1)

    cond do
      d == nil and s == [] -> 0.0
      s == [] -> d
      true -> %__MODULE__{data: d, shape: s}
    end
  end

  @spec contract_shape({[pos_integer], [non_neg_integer]}) :: Enum.t()
  defp contract_shape({s, a}) do
    Stream.with_index(s) |> Stream.reject(&(elem(&1, 1) in a)) |> Stream.map(&elem(&1, 0))
  end

  @spec replace(data | number, data | number, any) :: data
  defp replace(%{} = d1, d2, f) do
    Enum.into(d1, %{}, &{elem(&1, 0), replace(elem(&1, 1), d2, f)})
  end

  defp replace(d1, %{} = d2, f) do
    Enum.into(d2, %{}, &{elem(&1, 0), replace(d1, elem(&1, 1), f)})
  end

  defp replace(d1, d2, :prod), do: d1 * d2
  defp replace(d1, d2, :div), do: d1 / d2
  @spec combination(data, data, non_neg_integer) :: data | number | nil
  defp combination(d1, d2, 0), do: replace(d1, d2, :prod)

  defp combination(d1, d2, a) do
    Stream.filter(d1, &is_map_key(d2, elem(&1, 0)))
    |> Stream.map(fn {i, v1} -> combination(v1, d2[i], a - 1) end)
    |> reduce(fn v1, v2 -> merge(v1, v2, :add) end)
  end

  @spec reduce(Enum.t(), (Enum.element(), Enum.acc() -> Enum.acc())) :: Enum.acc()
  defp reduce(d, f) do
    try do
      Enum.reduce(d, f)
    rescue
      Enum.EmptyError -> nil
    end
  end

  @spec swap_axes(data, [non_neg_integer]) :: data
  defp swap_axes(d, []), do: d
  defp swap_axes(d, [i | a]), do: swap_axis(swap_axes(d, a), i - Enum.count(a, &(&1 > i)))
  @spec swap_axis(Enum.t(), non_neg_integer) :: Enum.t()
  defp swap_axis(d, 0), do: d

  defp swap_axis(d, a) do
    decomposite_by_axis(d, a)
    |> Enum.group_by(fn {_, {i, _}} -> i end, fn {i, {_, v}} -> {i, v} end)
    |> Enum.into(%{}, fn {i, v} -> {i, composite_by_axis(v)} end)
  end

  @spec decomposite_by_axis(Enum.t(), pos_integer) :: Enum.t()
  defp decomposite_by_axis(d, 1) do
    Stream.flat_map(d, fn {i, u} -> Stream.map(u, &{[i], &1}) end)
  end

  defp decomposite_by_axis(d, a) do
    Stream.flat_map(d, fn {i, u} ->
      Stream.map(decomposite_by_axis(u, a - 1), fn {j, v} -> {[i | j], v} end)
    end)
  end

  @spec composite_by_axis([{[pos_integer], data}]) :: data
  defp composite_by_axis([{[], d}]), do: d

  defp composite_by_axis(d) do
    Enum.group_by(d, fn {[i | _], _} -> i end, fn {[_ | i], v} -> {i, v} end)
    |> Enum.into(%{}, fn {i, v} -> {i, composite_by_axis(v)} end)
  end

  @doc """
  Divides all element of the tensor by the scalar.

      iex> #{__MODULE__}.divide(
      ...>   #{__MODULE__}.from_list([[2  , 3.5, -1.6,   8.2],
      ...>                            [1.1, 3.0,  0  , -12.1]]), 4)
      %#{__MODULE__}{data: %{1 => %{1 => 0.5  , 2 => 0.875, 3 => -0.4, 4 => 2.05  },
                             2 => %{1 => 0.275, 2 => 0.75 ,            4 => -3.025}}, shape: [2, 4]}
  """
  @spec divide(t, number) :: t
  def divide(%__MODULE__{data: nil} = t, s) when is_number(s) and s != 0, do: t

  def divide(%__MODULE__{data: d} = t, s) when is_number(s) and s != 0 do
    %{t | data: replace(d, s, :div)}
  end

  @doc """
  Returns a `n` × `n` tensor representing the kronecker delta.

      iex> #{__MODULE__}.kronecker_delta(3)
      %#{__MODULE__}{data: %{1 => %{1 => 1              },
                             2 => %{       2 => 1       },
                             3 => %{              3 => 1}}, shape: [3, 3]}
  """
  @spec kronecker_delta(pos_integer) :: t
  def kronecker_delta(n) when is_integer(n) and n > 0 do
    %__MODULE__{data: Enum.into(1..n, %{}, &{&1, %{&1 => 1}}), shape: [n, n]}
  end

  @doc """
  Performs the householder conversion.

  Returns a tuple of the converted vecter and the convertion tensor.
  Applying the conversion tensor to the given vector results to the converted vector.

      iex> #{__MODULE__}.householder(#{__MODULE__}.from_list([2  , 3.5, -1.6,   8.2]))
      {%#{__MODULE__}{data: %{1 => 9.276313923105448}, shape: [4]},
       %#{__MODULE__}{data: %{1 => %{1 => 0.21560288025811614, 2 => 0.3773050404517034  , 3 => -0.172482304206493, 4 => 0.8839718090582764},
                              2 => %{1 => 0.3773050404517034 , 2 => 0.8185114529779167  , 3 => 0.0829661929243809, 4 => -0.42520173873745204},
                              3 => %{1 => -0.172482304206493 , 2 => 0.0829661929243809  , 3 => 0.9620725975202831, 4 => 0.1943779377085495},
                              4 => %{1 => 0.8839718090582764 , 2 => -0.42520173873745204, 3 => 0.1943779377085495, 4 => 0.003813069243683853}}, shape: [4, 4]}}
  """
  @spec householder(t) :: {t, t}
  def householder(%__MODULE__{data: %{1 => x}, shape: [s]} = t) do
    dot = multiply(t, t, [{0, 0}])
    norm = if x < 0, do: :math.sqrt(dot), else: :math.sqrt(dot)
    v = divide(put_in(t[1], x - norm), dot * :math.sqrt(2 * abs(x - norm)))

    p =
      kronecker_delta(s)
      |> subtract(multiply(v, v) |> divide(multiply(v, v, [{0, 0}])) |> multiply(2))

    {%{t | data: %{1 => norm}}, p}
  end

  @doc """
  Returns a tensor consisting partial elements of the given tensor.

      iex>#{__MODULE__}.slice(#{__MODULE__}.from_list([[ 1,  2,  3],
      ...>                                             [ 4,  5,  6],
      ...>                                             [ 7,  8,  9],
      ...>                                             [10, 11, 12]]), 2..3)
      %#{__MODULE__}{data: %{1 => %{1 => 4, 2 => 5, 3 => 6},
                             2 => %{1 => 7, 2 => 8, 3 => 9}}, shape: [2, 3]}
      iex>#{__MODULE__}.slice(#{__MODULE__}.from_list([[ 1,  2,  3],
      ...>                                             [ 4,  5,  6],
      ...>                                             [ 7,  8,  9],
      ...>                                             [10, 11, 12]]), -2..-1)
      %#{__MODULE__}{data: %{1 => %{1 =>  7, 2 =>  8, 3 =>  9},
                             2 => %{1 => 10, 2 => 11, 3 => 12}}, shape: [2, 3]}
  """
  @spec slice(t, Range.t() | [Range.t]) :: t
  def slice(%__MODULE__{} = t, [r | s]), do: slice(t, s)
  def slice(%__MODULE__{shape: [s | _]} = t, m..n) when m < 0, do: slice(t, (s + m + 1)..n)
  def slice(%__MODULE__{shape: [s | _]} = t, m..n) when n < 0, do: slice(t, m..(s + n + 1))

  def slice(%__MODULE__{data: d, shape: [n | s]}, m.._ = r) do
    v = Stream.filter(d, &(elem(&1, 0) in r)) |> Enum.into(%{}, fn {i, u} -> {i - m + 1, u} end)
    %__MODULE__{data: v, shape: [Enum.count(r) | s]}
  end

  @doc """
  Tridiagonalize a symmetric 2-rank tensor.
  """
  def tridiagonalize(%__MODULE__{data: d, shape: [n, n]}) do

  end
end
