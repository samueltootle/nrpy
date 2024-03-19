ffmpeg -i z-%02d.png -vf palettegen palette.png
ffmpeg -framerate 2 -i z-%02d.png -i palette.png -lavfi paletteuse -r 15 z.gif
ffmpeg -framerate 2 -i y-%02d.png -i palette.png -lavfi paletteuse -r 15 y.gif
