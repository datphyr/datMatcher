color extractor from video files.
and color matcher to create LUT files.

it's 100% vibe coded, i have no idea how c++ works.
i was able to built it with msys2 ucrt64.
you need avcodec-62.dll from ffmpeg (can't upload cause 100mb limit on github).

g++ -o extract_colors.exe extract_colors.cpp $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale) -lpthread -O3 -march=native -ffast-math -flto
g++ -o match_colors.exe   match_colors.cpp   $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale) -lpthread -O3 -march=native -ffast-math -fopenmp -std=c++17
