mkdir -p zip
rm -rf zip/*.zip
7z a -tzip -mmt=on zip/metal.zip metal/*
7z a -tzip -mmt=on zip/paper.zip paper/*
7z a -tzip -mmt=on zip/plastic.zip plastic/*
7z a -tzip -mmt=on zip/glass.zip glass/*
7z a -tzip -mmt=on zip/others.zip others/*
