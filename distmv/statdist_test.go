// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package distmv

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func TestBhattacharyyaNormal(t *testing.T) {
	for cas, test := range []struct {
		am, bm  []float64
		ac, bc  *mat64.SymDense
		samples int
		tol     float64
	}{
		{
			am:      []float64{2, 3},
			ac:      mat64.NewSymDense(2, []float64{3, -1, -1, 2}),
			bm:      []float64{-1, 1},
			bc:      mat64.NewSymDense(2, []float64{1.5, 0.2, 0.2, 0.9}),
			samples: 100000,
			tol:     1e-2,
		},
	} {
		rnd := rand.New(rand.NewSource(1))
		a, ok := NewNormal(test.am, test.ac, rnd)
		if !ok {
			panic("bad test")
		}
		b, ok := NewNormal(test.bm, test.bc, rnd)
		if !ok {
			panic("bad test")
		}
		want := bhattacharryaSample(a.Dim(), test.samples, a, b)
		got := Bhattacharyya{}.DistNormal(a, b)
		if math.Abs(want-got) > test.tol {
			t.Errorf("Bhattacharyya mismatch, case %d: got %v, want %v", cas, got, want)
		}

		// Bhattacharyya should by symmetric
		got2 := Bhattacharyya{}.DistNormal(b, a)
		if math.Abs(got-got2) > 1e-14 {
			t.Errorf("Bhattacharyya distance not symmetric")
		}
	}
}

func TestBhattacharyyaUniform(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for cas, test := range []struct {
		a, b    *Uniform
		samples int
		tol     float64
	}{
		{
			a:       NewUniform([]Bound{{-3, 2}, {-5, 8}}, rnd),
			b:       NewUniform([]Bound{{-4, 1}, {-7, 10}}, rnd),
			samples: 100000,
			tol:     1e-2,
		},
	} {
		a, b := test.a, test.b
		want := bhattacharryaSample(a.Dim(), test.samples, a, b)
		got := Bhattacharyya{}.DistUniform(a, b)
		if math.Abs(want-got) > test.tol {
			t.Errorf("Bhattacharyya mismatch, case %d: got %v, want %v", cas, got, want)
		}
		// Bhattacharyya should by symmetric
		got2 := Bhattacharyya{}.DistUniform(b, a)
		if math.Abs(got-got2) > 1e-14 {
			t.Errorf("Bhattacharyya distance not symmetric")
		}
	}
}

// bhattacharryaSample finds an estimate of the Bhattacharrya coefficient through
// sampling.
func bhattacharryaSample(dim, samples int, l RandLogProber, r LogProber) float64 {
	lBhatt := make([]float64, samples)
	x := make([]float64, dim)
	for i := 0; i < samples; i++ {
		// Do importance sampling over a: \int sqrt(a*b)/a * a dx
		l.Rand(x)
		pa := l.LogProb(x)
		pb := r.LogProb(x)
		lBhatt[i] = 0.5*pb - 0.5*pa
	}
	logBc := floats.LogSumExp(lBhatt) - math.Log(float64(samples))
	return -logBc
}

func TestKullbackLieblerNormal(t *testing.T) {
	for cas, test := range []struct {
		am, bm  []float64
		ac, bc  *mat64.SymDense
		samples int
		tol     float64
	}{
		{
			am:      []float64{2, 3},
			ac:      mat64.NewSymDense(2, []float64{3, -1, -1, 2}),
			bm:      []float64{-1, 1},
			bc:      mat64.NewSymDense(2, []float64{1.5, 0.2, 0.2, 0.9}),
			samples: 10000,
			tol:     1e-2,
		},
	} {
		rnd := rand.New(rand.NewSource(1))
		a, ok := NewNormal(test.am, test.ac, rnd)
		if !ok {
			panic("bad test")
		}
		b, ok := NewNormal(test.bm, test.bc, rnd)
		if !ok {
			panic("bad test")
		}
		want := klSample(a.Dim(), test.samples, a, b)
		got := KullbackLeibler{}.DistNormal(a, b)
		if !floats.EqualWithinAbsOrRel(want, got, test.tol, test.tol) {
			t.Errorf("Case %d, KL mismatch: got %v, want %v", cas, got, want)
		}
	}
}

func TestKullbackLieblerUniform(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for cas, test := range []struct {
		a, b    *Uniform
		samples int
		tol     float64
	}{
		{
			a:       NewUniform([]Bound{{-5, 2}, {-7, 12}}, rnd),
			b:       NewUniform([]Bound{{-4, 1}, {-7, 10}}, rnd),
			samples: 100000,
			tol:     1e-2,
		},
	} {
		a, b := test.a, test.b
		want := klSample(a.Dim(), test.samples, a, b)
		got := KullbackLeibler{}.DistUniform(a, b)
		if math.Abs(want-got) > test.tol {
			t.Errorf("Bhattacharyya mismatch, case %d: got %v, want %v", cas, got, want)
		}
	}
}

// klSample finds an estimate of the Kullback-Liebler Divergence through sampling.
func klSample(dim, samples int, l RandLogProber, r LogProber) float64 {
	var klmc float64
	x := make([]float64, dim)
	for i := 0; i < samples; i++ {
		l.Rand(x)
		pa := l.LogProb(x)
		pb := r.LogProb(x)
		klmc += pa - pb
	}
	return klmc / float64(samples)
}
