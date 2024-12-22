# CV_Final_Proj

chmod +x /home/yw4142/starter_scripts/gpu.sh <br>
/home/yw4142/starter_scripts/gpu.sh

/home/yw4142/3d/CV_Final_Proj/spann3r

/home/yw4142/starter_scripts/sing_rw.sh

/home/yw4142/starter_scripts/sing_ro.sh
mamba activate spann3r

curl ifconfig.me
216.165.12.11

O3D_USE_OPENGL=1 WEBRTC_IP=0.0.0.0 WEBRTC_PORT=8888 python demo.py --kf_every 10 --vis --vis_cam
PYOPENGL_PLATFORM=egl OPEN3D_RENDER_OFFSCREEN=1 python demo.py --demo_path /vast/yw4142/lh_data/Portland_Hotel --kf_every 10 --vis --vis_cam
python demo.py --demo_path /vast/yw4142/lh_data/brown_cogsci_7/brown_cogsci_7 --kf_every 10

cd /home/yw4142/3d/CV_Final_Proj/spann3r

wget -r -np -nH --cut-dirs=3 -R index.html https://sun3d.cs.princeton.edu/data/harvard_tea_1/hv_tea1_2/image/

python evaluator.py --gt /vast/yw4142/checkpoints/spann3r/checkpoints/output/demo/brown_cogsci_7/time.ply --pred /vast/yw4142/checkpoints/spann3r/checkpoints/output/demo/brown_cogsci_7/brown_cogsci_7_conf0.001_fullmem.ply --out /vast/yw4142/checkpoints/spann3r/checkpoints

scp gr:/vast/yw4142/checkpoints/spann3r/checkpoints/output/demo/brown_cogsci_4/brown_cogsci_4_conf0.001.ply /Users/yw511/Desktop/NYU_MSCS/1-CV/project/pointMaps