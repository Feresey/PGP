package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHexArray(t *testing.T) {
	h := newHexArrayTool(&command{mode: "encode"})
	h.FirstLength = true

	t.Run("encode", func(t *testing.T) {
		err := h.RunE(nil, nil)
		require.NoError(t, err)
	})
	h.mode = "decode"
	t.Run("encode", func(t *testing.T) {
		err := h.RunE(nil, nil)
		require.NoError(t, err)
	})
}
