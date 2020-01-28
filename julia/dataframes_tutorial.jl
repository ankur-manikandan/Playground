using DataFrames

df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"])
println(df)
println(df.A)


df = DataFrame(A = String[], B = String[], C = String[])
col1 = ['a':'z';]
col2 = ['A':'Z';]
col3 = ['0':'9';]

N = 10^6

for _ in 1:N
    push!(df, (String(rand(col1, 1)), String(rand(col2, 1)),
    String(rand(col3, 1))))
end
