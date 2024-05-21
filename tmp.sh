VALID_OPERATORS=(
    "+"
    "-"
    "*"
    "/"
    "**2+"
    "**3+"
    "x**2+y**2_mod_97"
    "x**2+y**2+x*y_mod_97"
    "x**2+y**2+x*y+x_mod_97"
    "x**3+x*y_mod_97"
    "x**3+x*y**2+y_mod_97"
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97"
    "s5"
    "s5conj"
    "s5aba"
    "+*"
    "+-"
    "pfactor"
    "2x"
    "x**3"
    "2x+1"
    "x+11"
    "sort"
    "reverse"
    "copy"
    "interleaved_halves"
    "reverse_pool"
    "k_shift"
    "random_swaps"
    "idx_add"
    "caesarcipher"
    "permutev1"
    "permutev2"
    "permutev3"
    "strdeletev1"
    "strdeletev2"
    "caesarcipher_permutev1"
)



# Clear the output file
> out.txt

# Run the python command for each operator and append the output to out.txt
for operator in "${VALID_OPERATORS[@]}"; do
    echo "Running for operator: $operator"
    python main.py exp=ff tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 debug=True math_operator="$operator" >> out.txt
done
