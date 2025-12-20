
conda activate serl2

cd /home/magictavern/projects/openpi
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build