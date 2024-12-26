// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Neural is a neural network
type Neural struct {
	Set    tf64.Set
	Others tf64.Set
	L1     tf64.Meta
	L2     tf64.Meta
	Loss   tf64.Meta
}

// Learn learn a neural network
func Learn(txts []TXT) Neural {
	rng := rand.New(rand.NewSource(1))
	set := tf64.NewSet()
	set.Add("w1", 256, 1024)
	set.Add("b1", 1024)
	set.Add("w2", 1024, 256)
	set.Add("b2", 256)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	others.Add("input", 256)
	others.Add("output", 256)

	for i := range others.Weights {
		w := others.Weights[i]
		w.X = w.X[:cap(w.X)]
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
	l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
	loss := tf64.Quadratic(l2, others.Get("output"))

	last := 0.0
	points := make(plotter.XYs, 0, 8)
	fmt.Println("learning:", len(txts))
	for i := 0; i < len(txts); i++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		others.Zero()
		index := rng.Intn(len(txts))
		input := others.ByName["input"].X
		sum := 0.0
		for _, v := range txts[index].Vector {
			sum += float64(v)
		}
		for j := range input {
			input[j] = float64(txts[index].Vector[j]) / sum
		}
		output := others.ByName["output"].X
		for j := range output {
			output[j] = 0
		}
		output[txts[index].Symbol] = 1

		set.Zero()
		cost := tf64.Gradient(loss).X[0]
		last = cost

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		if i%1024 == 0 {
			fmt.Println(cost)
		}
	}

	err := set.Save("set.db", last, len(txts))
	if err != nil {
		panic(err)
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
	return Neural{
		Set:    set,
		Others: others,
		L1:     l1,
		L2:     l2,
		Loss:   loss,
	}
}

// Inference performs inference of the neural network
func (n *Neural) Inference(input [256]float64) int {
	symbol, max := 0, 0.0
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L2(func(a *tf64.V) bool {
		for i, v := range a.X {
			if v > max {
				max, symbol = v, i
			}
		}
		return true
	})

	return symbol
}
