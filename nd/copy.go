package nd

// Copy copies src into dst and returns dst.
func Copy(dst, src *Iter) {
	if len(src.stri) != len(dst.stri) {
		panic("dimensions mismatch")
	}

	if src.size != dst.size {
		panic("iterators not compatible. must be same size")
	}

	for k := 0; k < dst.size; k++ {
		ss, ds := ZipInd(src, dst, k)
		sv, dv := src.x[ss:], dst.x[ds:]
		j, k := 0, 0
		for n := dst.rn; n != 0; n-- {
			dv[j] = sv[k]
			j += dst.rs
			k += src.rs
		}
	}
}

// Clone returns a new Ndarray with a copy of the data in array.
func (array *Ndarray) Clone() *Ndarray {
	res := Zeroslike(array)
	Copy(res.Iter(), array.Iter())
	return res
}
