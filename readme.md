# Official code and data for{Learning the Boundary of the Solvable: Alignment of LLMs with Problem Unsolvability}

Ensuring LLM reliability requires not only solving complex problems but also recognizing when a problem is unsolvable. Current models often struggle to distinguish \textit{objective unsolvability} (inherent contradictions in the problem) from \textit{subjective capability limitations} (problems beyond the model’s competence), which leads to hallucinations and overconfidence. To address this, we propose UnsolvableQA and UnsolvableRL to solve feasible problems, detect inherent contradictions, and prudently refuse tasks beyond capability. Specifically, we construct \textbf{UnsolvableQA}, a dataset of paired solvable and unsolvable instances derived via a dual-track methodology: programmatic generation for logic puzzles and a novel ``\textbf{Reverse Construction}'' method that injects contradictions into valid reasoning chains for mathematics. Building on this dataset, we introduce UnsolvableRL, a reinforcement learning framework with three reward components jointly accounting for accuracy, unsolvability, and difficulty. Empirical results show that our approach achieves near-perfect unsolvability detection while also improving accuracy on solvable tasks. Crucially, we identify {Capability Collapse}, demonstrating that explicit exposure to unsolvable data is indispensable for preventing models from becoming systematically overconfident.

The data generation code for AIME and Maze is currently being prepared. We have provided the unsolvable data generation code for Game24, Hamiltonian Cycle, Hamiltonian Path, and Hitori.


If you find our work inspiring or helpful, please cite:

```bibtex
@misc{peng2025learningboundarysolvabilityaligning,
      title={Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems}, 
      author={Dengyun Peng and Qiguang Chen and Bofei Liu and Jiannan Guan and Libo Qin and Zheng Yan and Jinhao Liu and Jianshu Zhang and Wanxiang Che},
      year={2025},
      eprint={2512.01661},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.01661}, 
}
```
