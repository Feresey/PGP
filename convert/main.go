package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

type command struct {
	input, output, mode string
}

func New() *cobra.Command {
	c := new(command)
	root := &cobra.Command{}
	convert := &cobra.Command{
		Use: "convert",
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			switch c.mode {
			case "encode", "decode":
				return nil
			case "":
				return errors.New("mode flag required")
			default:
				return fmt.Errorf("unknown mode: %q. expected encode or decode", c.mode)
			}
		},
	}
	c.flags(convert, nil)

	convert.AddCommand(c.imageCommand(), c.hexCommand(), c.hexArrayCommand(), c.pixelsCommand())

	root.AddCommand(convert, newTableCommand())

	return root
}

func (c *command) flags(cmd *cobra.Command, args []string) {
	flags := cmd.PersistentFlags()
	flags.StringVarP(&c.input, "input", "i", "", "input file")
	flags.StringVarP(&c.output, "output", "o", "", "output file")
	flags.StringVar(&c.mode, "mode", "", "convert mode")
}

func (c *command) imageCommand() *cobra.Command {
	return &cobra.Command{
		Use:  "image",
		RunE: newImageTool(c).RunE,
	}
}

func (c *command) hexCommand() *cobra.Command {
	return &cobra.Command{
		Use:  "hex",
		RunE: newHexTool(c).RunE,
	}
}

func (c *command) hexArrayCommand() *cobra.Command {
	ha := newHexArrayTool(c)
	cmd := &cobra.Command{
		Use:  "hex-array",
		RunE: ha.RunE,
	}
	ha.flags(cmd)

	return cmd
}

func (c *command) pixelsCommand() *cobra.Command {
	return &cobra.Command{
		Use:  "pixels",
		RunE: newPixelsTool(c).RunE,
	}
}

func main() {
	cmd := New()
	if err := cmd.Execute(); err != nil {
		println(err)
		os.Exit(1)
	}
}
