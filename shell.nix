{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
	packages = [
		pkgs.python313
		pkgs.python313Packages.pip
		pkgs.virtualenv
	];

	env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ 
		pkgs.stdenv.cc.cc.lib
		pkgs.libz
		pkgs.libGL
		pkgs.glibc
		pkgs.glib
	];

	shellHook = ''
		source ./venv/bin/activate
+	'';
}
