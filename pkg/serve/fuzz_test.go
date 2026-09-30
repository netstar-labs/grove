package serve

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// isWithinDir reports whether path is dir or a descendant of it.
func isWithinDir(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// FuzzTrainBody feeds arbitrary bytes as the raw POST /train body — the
// exact trust boundary an unauthenticated network peer reaches (see this
// pass's NumClass/Rounds finding, which lived here). The only invariant:
// never panic, never hang; a decode or validation error is an ordinary,
// expected response.
func FuzzTrainBody(f *testing.F) {
	f.Add([]byte(`{"params":{"objective":"binary","rounds":5},"features":[[1],[2]],"labels":[0,1]}`))
	f.Add([]byte(`{"params":{"objective":"multiclass","num_class":3},"features":[[1]],"labels":[0]}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`not json`))
	f.Add([]byte(`{"params":{"num_class":-1}}`))
	h := New(f.TempDir()).Handler()
	f.Fuzz(func(t *testing.T, body []byte) {
		req := httptest.NewRequest(http.MethodPost, "/train", bytes.NewReader(body))
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
	})
}

// FuzzLoadName feeds arbitrary bytes as the ?name= query parameter to
// POST /load — the path-validation trust boundary (path traversal, invalid
// characters). Never panic, never escape the model directory.
func FuzzLoadName(f *testing.F) {
	f.Add("model")
	f.Add("../../etc/passwd")
	f.Add("")
	f.Add(".")
	f.Add("..")
	f.Add("a/b")
	s := New(f.TempDir())
	f.Fuzz(func(t *testing.T, name string) {
		path, err := s.path(name)
		if err == nil && !isWithinDir(path, s.dir) {
			t.Fatalf("path(%q) = %q escapes model dir %q", name, path, s.dir)
		}
	})
}
