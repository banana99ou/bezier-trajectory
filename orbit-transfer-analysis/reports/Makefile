# FMCL Technical Reports - Build System
# Usage:
#   make all       - 전체 보고서 컴파일
#   make A1        - 개별 보고서 컴파일
#   make A_physics - 주제별 전체 컴파일
#   make clean     - 임시 파일 삭제
#   make distclean - PDF 포함 전체 삭제

LATEX = pdflatex
BIBTEX = bibtex
GS = gs
LATEXFLAGS = -interaction=nonstopmode -halt-on-error
GSFLAGS = -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
          -dNOPAUSE -dQUIET -dBATCH \
          -dPDFSETTINGS=/prepress \
          -dEmbedAllFonts=true -dSubsetFonts=true \
          -dCompressFonts=true -dCompressPages=true

# 임시 파일 확장자
AUX_EXTS = aux log bbl blg out toc lof lot nav snm vrb fdb_latexmk fls synctex.gz

# 서브디렉토리별 보고서 정의
A_DIR = A_physics
B_DIR = B_discretization
C_DIR = C_optimization
D_DIR = D_classification

A_REPORTS = A1_coordinate_systems A2_environmental_models A3_hohmann_benchmark
B_REPORTS = B1_hermite_simpson B2_lgl_pseudospectral B3_multiphase_collocation
C_REPORTS = C1_nlp_formulation C2_initial_guess_strategy
D_REPORTS = D1_gp_classifier_impl D2_active_learning_algorithm D3_latin_hypercube_sampling D4_peak_detection

ALL_REPORTS = $(A_REPORTS) $(B_REPORTS) $(C_REPORTS) $(D_REPORTS)

# 개별 보고서의 디렉토리 매핑
dir_of = $(if $(filter A%,$1),$(A_DIR),$(if $(filter B%,$1),$(B_DIR),$(if $(filter C%,$1),$(C_DIR),$(D_DIR))))

# PDF 경로 생성
A_PDFS = $(addprefix $(A_DIR)/,$(addsuffix .pdf,$(A_REPORTS)))
B_PDFS = $(addprefix $(B_DIR)/,$(addsuffix .pdf,$(B_REPORTS)))
C_PDFS = $(addprefix $(C_DIR)/,$(addsuffix .pdf,$(C_REPORTS)))
D_PDFS = $(addprefix $(D_DIR)/,$(addsuffix .pdf,$(D_REPORTS)))
ALL_PDFS = $(A_PDFS) $(B_PDFS) $(C_PDFS) $(D_PDFS)

# 단축 타겟 - 개별 보고서
A1: $(A_DIR)/A1_coordinate_systems.pdf
A2: $(A_DIR)/A2_environmental_models.pdf
A3: $(A_DIR)/A3_hohmann_benchmark.pdf
B1: $(B_DIR)/B1_hermite_simpson.pdf
B2: $(B_DIR)/B2_lgl_pseudospectral.pdf
B3: $(B_DIR)/B3_multiphase_collocation.pdf
C1: $(C_DIR)/C1_nlp_formulation.pdf
C2: $(C_DIR)/C2_initial_guess_strategy.pdf
D1: $(D_DIR)/D1_gp_classifier_impl.pdf
D2: $(D_DIR)/D2_active_learning_algorithm.pdf
D3: $(D_DIR)/D3_latin_hypercube_sampling.pdf
D4: $(D_DIR)/D4_peak_detection.pdf

# 단축 타겟 - 주제별
A_physics: $(A_PDFS)
B_discretization: $(B_PDFS)
C_optimization: $(C_PDFS)
D_classification: $(D_PDFS)

.PHONY: all clean distclean A1 A2 A3 B1 B2 B3 C1 C2 D1 D2 D3 D4 \
        A_physics B_discretization C_optimization D_classification

all: $(ALL_PDFS)
	@echo "=== 전체 보고서 컴파일 완료 (12편) ==="
	@$(MAKE) clean-aux

# 서브디렉토리 내에서 컴파일하는 패턴 규칙
$(A_DIR)/%.pdf: $(A_DIR)/%.tex references.bib reusv.sty
	cd $(A_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(A_DIR) && $(BIBTEX) $* > /dev/null 2>&1 || true
	cd $(A_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(A_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	@cd $(A_DIR) && $(GS) $(GSFLAGS) -sOutputFile=$*.pdf.tmp $*.pdf > /dev/null 2>&1 && mv $*.pdf.tmp $*.pdf || true
	@echo "[완료] $(A_DIR)/$*.pdf"

$(B_DIR)/%.pdf: $(B_DIR)/%.tex references.bib reusv.sty
	cd $(B_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(B_DIR) && $(BIBTEX) $* > /dev/null 2>&1 || true
	cd $(B_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(B_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	@cd $(B_DIR) && $(GS) $(GSFLAGS) -sOutputFile=$*.pdf.tmp $*.pdf > /dev/null 2>&1 && mv $*.pdf.tmp $*.pdf || true
	@echo "[완료] $(B_DIR)/$*.pdf"

$(C_DIR)/%.pdf: $(C_DIR)/%.tex references.bib reusv.sty
	cd $(C_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(C_DIR) && $(BIBTEX) $* > /dev/null 2>&1 || true
	cd $(C_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(C_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	@cd $(C_DIR) && $(GS) $(GSFLAGS) -sOutputFile=$*.pdf.tmp $*.pdf > /dev/null 2>&1 && mv $*.pdf.tmp $*.pdf || true
	@echo "[완료] $(C_DIR)/$*.pdf"

$(D_DIR)/%.pdf: $(D_DIR)/%.tex references.bib reusv.sty
	cd $(D_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(D_DIR) && $(BIBTEX) $* > /dev/null 2>&1 || true
	cd $(D_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	cd $(D_DIR) && $(LATEX) $(LATEXFLAGS) $*.tex > /dev/null 2>&1
	@cd $(D_DIR) && $(GS) $(GSFLAGS) -sOutputFile=$*.pdf.tmp $*.pdf > /dev/null 2>&1 && mv $*.pdf.tmp $*.pdf || true
	@echo "[완료] $(D_DIR)/$*.pdf"

clean-aux:
	@for dir in $(A_DIR) $(B_DIR) $(C_DIR) $(D_DIR); do \
		for ext in $(AUX_EXTS); do \
			rm -f $$dir/*.$$ext; \
		done; \
	done

clean: clean-aux
	@echo "임시 파일 삭제 완료"

distclean: clean
	rm -f $(ALL_PDFS)
	@echo "PDF 포함 전체 삭제 완료"
