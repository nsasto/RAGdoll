# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.go
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodejs_20
    pkgs.nodePackages.nodemon
    pkgs.libGL
    pkgs.python311Packages.pytest
  ];

  # Sets environment variables in the workspace
  env = {
    VENV_DIR = ".venv";
  };

  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # "vscodevim.vim"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # web = {
        #   # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
        #   # and show it in IDX's web preview panel
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     # Environment variables to set for your server
        #     PORT = "$PORT";
        #   };
        # };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      onCreate = {
        create-venv = ''
          # Create the virtual environment if it doesn't exist
          if [ ! -d "$VENV_DIR" ]; then
            python -m venv $VENV_DIR
          fi

          # Activate the virtual environment
          source $VENV_DIR/bin/activate

          # Install dependencies if requirements.txt exists
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi
        '';
      };

      onStart = {
        activate-venv = ''
          # Activate the virtual environment
          source $VENV_DIR/bin/activate
        '';
      };
    };
  };
}
