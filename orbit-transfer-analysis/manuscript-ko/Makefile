TEX      = pdflatex
BIB      = bibtex
GS       = gs
MAIN     = manuscript
FLAGS    = -interaction=nonstopmode

# 임시 파일 확장자 목록
AUX_EXTS = aux bbl blg log out fdb_latexmk fls synctex.gz abs

.PHONY: all pdf bib clean open watch quick view distclean compress

# 기본 타겟: 전체 빌드 → 압축 → 임시 파일 삭제
all: pdf

pdf:
	$(TEX) $(FLAGS) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(FLAGS) $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex
	@$(MAKE) compress
	@$(MAKE) clean
	@echo "==> $(MAIN).pdf 생성 완료 (한국어)"

# Ghostscript PDF 압축 (원본보다 작을 때만 적용)
compress:
	@BEFORE=$$(wc -c < $(MAIN).pdf | tr -d ' '); \
	$(GS) -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
	      -dNOPAUSE -dBATCH -dQUIET \
	      -dPDFSETTINGS=/prepress \
	      -sOutputFile=$(MAIN)_compressed.pdf $(MAIN).pdf; \
	AFTER=$$(wc -c < $(MAIN)_compressed.pdf | tr -d ' '); \
	if [ "$$AFTER" -lt "$$BEFORE" ]; then \
	  mv $(MAIN)_compressed.pdf $(MAIN).pdf; \
	  echo "==> PDF 압축 적용: $$BEFORE → $$AFTER bytes"; \
	else \
	  rm -f $(MAIN)_compressed.pdf; \
	  echo "==> PDF 압축 생략 (원본 $$BEFORE bytes ≤ 압축 $$AFTER bytes)"; \
	fi

# bibtex만 실행 (참고문헌 변경 시)
bib:
	$(TEX) $(FLAGS) $(MAIN).tex
	$(BIB) $(MAIN)

# 빠른 빌드 (pdflatex 1회만, 압축/정리 없음)
quick:
	$(TEX) $(FLAGS) $(MAIN).tex

# PDF 열기 (macOS)
open: pdf
	open $(MAIN).pdf

# PDF만 열기 (빌드 없이)
view:
	open $(MAIN).pdf

# 임시 파일 삭제
clean:
	@rm -f $(foreach ext,$(AUX_EXTS),$(MAIN).$(ext))

# 임시 파일 + PDF 모두 삭제
distclean: clean
	rm -f $(MAIN).pdf
	@echo "==> PDF 삭제 완료"

# latexmk 감시 모드 (파일 변경 시 자동 빌드)
watch:
	latexmk -pvc -pdf -interaction=nonstopmode $(MAIN).tex
