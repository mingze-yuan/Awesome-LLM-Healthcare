# Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review

## 🔔 News
<!-- - 💥 [2023/09/15] Our survey is released! See [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864) for the paper! -->
- ✨ [2023/11/03] We create this repository to maintain a paper list on LLM-based agents. More papers are coming soon!

<div align=center><img src="./assets/main.png" width="90%" /></div>

## 🌟 Introduction
With the rapid development of artificial intelligence, large language models (LLMs) have shown promising capabilities in mimicking human-level language comprehension and reasoning. This has sparked significant interest in applying LLMs to enhance various aspects of healthcare, ranging from medical education to clinical decision support. However, medicine involves multifaceted data modalities and nuanced reasoning skills, presenting challenges for integrating LLMs. 

This repository provides a comprehensive review on the applications and implications of LLMs in medicine. It begins by examining the fundamental applications of **general-purpose and specialized LLMs**, demonstrating their utilities in knowledge retrieval, research support, clinical workflow automation, and diagnostic assistance. Recognizing the inherent multimodality of medicine, the review then focuses on **multimodal LLMs**, investigating their ability to process diverse data types like medical imaging and EHRs to augment diagnostic accuracy. To address LLMs’ limitations regarding personalization and complex clinical reasoning, the paper explores the emerging development of **LLM-powered autonomous agents for healthcare**. Furthermore, it summarizes the evaluation methodologies for assessing LLMs’ reliability and safety in medical contexts. Overall, this review offers an extensive analysis on the transformative potential of LLMs in modern medicine. It also highlights the pivotal need for continuous optimizations and ethical oversight before these models can be effectively integrated into clinical practice.

**We sincerely value all contributions, whether through pull requests, issue reports, emails, or other forms of communication.**

## Specialized Medical LLMs
- [2022/10] **Health system-scale language models are all-purpose prediction engines** *Lavender Yao Jiang et al.* [[paper](https://www.nature.com/articles/s41586-023-06160-y)] [[code](https://github.com/nyuolab/NYUTron)] 
- [2022/12] **Large language models encode clinical knowledge** *Karan Singhal et al. Nature.* [[paper](https://www.nature.com/articles/s41586-023-06291-2)] [[code](https://huggingface.co/google/flan-t5-xl)]
- [2022/12] **A large language model for electronic health records** *Xi Yang et al. npj Digital Medicine.* [[paper](https://www.nature.com/articles/s41746-022-00742-2)] [[code](https://github.com/uf-hobi-informatics-lab/GatorTron)]
- [2023/03] **ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge** *Yunxiang Li et al. Cureus.* [[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10364849/)] [[code](https://github.com/Kent0n-Li/ChatDoctor)]
- [2023/03] **ChatGLM-Med** *Haochun Wang et al.* [[code](https://github.com/SCIR-HI/Med-ChatGLM)]
- [2023/03] **Palmyra-Large Parameter Autoregressive Language Model** *Writer Engineering team.* [[code](https://huggingface.co/Writer/palmyra-med-20b)]
- [2023/04] **Doctor Dignity** *Siraj Raval et al.* [[code](https://github.com/llSourcell/Doctor-Dignity)]
- [2023/04] **HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge** *Haochun Wang et al. arXiv.* [[paper](https://arxiv.org/abs/2304.06975)] [[code](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)]
- [2023/04] **PMC-LLaMA: Towards Building Open-source Language Models for Medicine** *Chaoyi Wu et al. arXiv.* [[paper](https://arxiv.org/abs/2304.14454)] [[code](https://github.com/chaoyi-wu/PMC-LLaMA)]
- [2023/04] **DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task** *Honglin Xiong et al.* [[paper](https://arxiv.org/abs/2304.01097)] [[code](https://github.com/xionghonglin/DoctorGLM)]
- [2023/04] **ChatMed: A Chinese Medical Large Language Model** *Wei Zhu et al.* [[code](https://github.com/michael-wzhu/ChatMed)]
- [2023/04] **BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT** *Yirong Chen et al.* arXiv. [[paper](https://arxiv.org/abs/2310.15896)] [[code](https://github.com/scutcyr/BianQue)]
- [2023/04] **MedAlpaca--An Open-Source Collection of Medical Conversational AI Models and Training Data** *Tianyu Han et al.* arXiv:2304.08247 [[paper](https://arxiv.org/abs/2304.08247)] [[code](https://github.com/kbressem/medAlpaca)]
- [2023/05] **HuatuoGPT, towards Taming Language Model to Be a Doctor** *Hongbo Zhang et al. arXiv.* [[paper](https://arxiv.org/abs/2305.15075)] [[code](https://github.com/FreedomIntelligence/HuatuoGPT)]
- [2023/05] **QiZhenGPT: An Open Source Chinese Medical Large Language Model** *Yao Chang et al.* [[code](https://github.com/CMKRG/QiZhenGPT)]
- [2023/05] **CMLM-ZhongJing: Large Language Model is Good Story Listener** *Yanlan Kang et al.* [[code](https://github.com/pariskang/CMLM-ZhongJing)]
- [2023/05] **Towards Expert-Level Medical Question Answering with Large Language Models** *Karan Singhal et al. arXiv.* [[paper](https://arxiv.org/abs/2305.09617)]
- [2023/05] **Clinfo.AI: Answer Clinical Questions Grounded in Medical Literature** *Alejandro Lozano et al.* [[code](https://github.com/som-shahlab/Clinfo.AI)]
- [2023/06] **ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation** *Guangyu Wang et al.* arXiv. [[paper](https://arxiv.org/abs/2306.09968)]
- [2023/06] **MedicalGPT: Training Medical GPT Model** *Ming Xu et al.* [[code](https://github.com/shibing624/MedicalGPT)]
- [2023/06] **Radiology-GPT: A Large Language Model for Radiology** *Zhengliang Liu et al.* [[paper](https://arxiv.org/abs/2306.08666)] [[code](https://huggingface.co/spaces/allen-eric/radiology-gpt)] 
- [2023/06] **ShenNong-TCM: A Traditional Chinese Medicine Large Language Model** *Wei Zhu et al.* [[code](https://github.com/michael-wzhu/ShenNong-TCM-LLM)]
- [2023/06] **Sunsimiao: Chinese Medicine LLM** *Xin Yan et al.* [[code](https://github.com/thomas-yanxin/Sunsimiao)]
- [2023/06] **PULSE** *OpenMedLab*. [[code](https://github.com/openmedlab/PULSE)]
- [2023/06] **TCMLLM** *Xuezhong Zhou et al.* [[code](https://github.com/2020MEAI/TCMLLM)]
- [2023/07] **MING: A Chinese Medical Consultation Large Model** *Yusheng Liao et al.* [[code](https://github.com/MediaBrain-SJTU/MING)]
- [2023/07] **HuangDi: A Generative Large Language Model for Ancient Chinese Medical Texts** *Jundong Zhang et al.* [[code](https://github.com/Zlasejd/HuangDi)]
- [2023/08] **CareGPT: Medical LLM, Open Source Driven for a Healthy Future** *Rongsheng Wang et al.* [[code](https://github.com/WangRongsheng/CareGPT)]
- [2023/08] **DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation** *Zhijie Bao et al.* [[code](https://github.com/FudanDISC/DISC-MedLLM)] [[paper](https://arxiv.org/abs/2308.14346)]
- [2023/08] **Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue** *Songhua Yang et al.* arXiv. [[paper](https://arxiv.org/abs/2308.03549)] [[code](https://github.com/SupritYoung/Zhongjing)]
- [2023/09] **HealGPT** *GR-Tech* [[code](https://github.com/TONYCHANBB/HealGPT)] [[demo](http://heal-gpt.cn/)]
- [2023/09] **Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks** *Taiyi-Team.* [[code](https://github.com/DUTIR-BioNLP/Taiyi-LLM)]

## Multimodal LLMs in Medicine
- [2023/04] **Visual Med-Alpaca: A Parameter-Efficient Biomedical LLM with Visual Capabilities** *Chang Shu et al.* [[blog](https://cambridgeltl.github.io/visual-med-alpaca/)] [[code](https://github.com/cambridgeltl/visual-med-alpaca)]
- [2023/04] **SkinGPT-4: An Interactive Dermatology Diagnostic System with Visual Large Language Model** *Juexiao Zhou et al. arXiv.* [[paper](https://arxiv.org/abs/2304.10691)] [[code](https://github.com/JoshuaChou2018/SkinGPT-4)]
- [2023/05] **XrayGLM: The first Chinese Medical Multimodal Model that Chest Radiographs Summarization** *Rongsheng Wang et al.* [[code](https://github.com/WangRongsheng/XrayGLM)]
- [2023/05] **PathAsst: Redefining Pathology through Generative Foundation AI Assistant for Pathology** *Yuxuan Sun et al. arXiv.* [[paper](https://arxiv.org/abs/2305.15072)] [[code](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology)]
- [2023/05] **BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks** *Kai Zhang et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17100)] [[code](https://github.com/taokz/BiomedGPT)]
- [2023/05] **PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering** *Xiaoman Zhang et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10415)] [[code](https://github.com/xiaoman-zhang/PMC-VQA)]
- [2023/06] **LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day** *Chunyuan Li et al. arXiv.* [[paper](https://arxiv.org/abs/2306.00890)] [[code](https://github.com/microsoft/LLaVA-Med/blob/main/README.md)]
- [2023/06] **XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models** *Omkar Thawkar et al. arXiv.* [[paper](https://arxiv.org/abs/2306.07971)] [[code](https://github.com/mbzuai-oryx/XRayGPT)]
- [2023/06] **XrayPULSE** *OpenMedLab.* [[code](https://github.com/openmedlab/XrayPULSE)]
- [2023/06] **Lmflow: An extensible toolkit for finetuning and inference of large foundation models** *Shizhe Diao et al.* [[code](https://github.com/OptimalScale/LMFlow)] [[paper](https://arxiv.org/abs/2306.12420)] [[blog](https://optimalscale.github.io/LMFlow/)]
- [2023/07] **CephGPT-4: An Interactive Multimodal Cephalometric Measurement and Diagnostic System with Visual Large Language Model** *Lei Ma et al. arXiv.* [[paper](https://arxiv.org/abs/2307.07518)]
- [2023/07] **Towards Generalist Biomedical AI** *Tao Tu et al. arXiv.* [[paper](https://arxiv.org/abs/2307.14334)] [[code](https://github.com/kyegomez/Med-PaLM)]
- [2023/07] **Med-Flamingo: a Multimodal Medical Few-shot Learner** *Michael Moor et al. arXiv.* [[paper](https://arxiv.org/abs/2307.15189)] [[code](https://github.com/snap-stanford/med-flamingo)]
- [2023/07] **Multimodal LLMs for health grounded in individual-specific data** *Anastasiya Belyaeva et al. arXiv.* [[paper](https://arxiv.org/abs/2307.09018)] [[blog](https://blog.research.google/2023/08/multimodal-medical-ai.html)]
- [2023/08] **BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine** *Yizhen Luo et al. arXiv.* [[paper](https://arxiv.org/abs/2308.09442)] [[code](https://github.com/PharMolix/OpenBioMed)]
- [2023/08] **Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data** *Chaoyi Wu et al. arXiv.* [[paper](https://arxiv.org/abs/2308.02463)] [[code](https://github.com/chaoyi-wu/RadFM)]
- [2023/08] **ELIXR: Towards a general purpose X-ray artificial intelligence system through alignment of large language models and radiology vision encoders** *Shawn Xu et al. arXiv.* [[paper](https://arxiv.org/abs/2308.01317)]

## Acknowledgement
We have structured our repository by drawing inspiration from the substantial work of repositories such as [LLM-Agent-Paper-List](https://github.com/WooooDyy/LLM-Agent-Paper-List/tree/main), [CareGPT](https://github.com/WangRongsheng/CareGPT), and insights from [RadLLM](https://arxiv.org/pdf/2307.13693.pdf). We extend our sincere gratitude to their contributions.