package nd

// Axpy computes the operation y += a*x
func Axpy(a float64, x, y *Iter) {
	if len(x.stri) != len(y.stri) {
		panic("dimensions mismatch")
	}

	if x.size != y.size {
		panic("iterators not compatible. must be same size")
	}

	for k := 0; k < y.size; k++ {
		xs, ys := ZipInd(x, y, k)
		xv, yv := x.x[xs:], y.x[ys:]
		i, j := 0, 0
		for n := y.rn; n != 0; n-- {
			yv[j] += a * xv[i]
			i += x.rs
			j += y.rs
		}
	}
}
