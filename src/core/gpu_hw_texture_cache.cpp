// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_hw_texture_cache.h"
#include "gpu.h"

#include "util/gpu_device.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/hash_combine.h"
#include "common/log.h"
#include "common/small_string.h"

#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

Log_SetChannel(GPUTextureCache);

[[maybe_unused]] ALWAYS_INLINE static TinyString RectToString(const Common::Rectangle<u32>& rc)
{
  return TinyString::from_format("{},{} => {},{} ({}x{})", rc.left, rc.top, rc.right, rc.bottom, rc.GetWidth(),
                                 rc.GetHeight());
}

ALWAYS_INLINE_RELEASE static void ListPrepend(GPUTextureCache::SourceList* list, GPUTextureCache::Source* item,
                                              GPUTextureCache::SourceListNode* item_node)
{
  item_node->ref = item;
  item_node->list = list;
  item_node->prev = nullptr;
  if (list->tail)
  {
    item_node->next = list->head;
    list->head->prev = item_node;
    list->head = item_node;
  }
  else
  {
    item_node->next = nullptr;
    list->head = item_node;
    list->tail = item_node;
  }
}

ALWAYS_INLINE_RELEASE static void ListAppend(GPUTextureCache::SourceList* list, GPUTextureCache::Source* item,
                                             GPUTextureCache::SourceListNode* item_node)
{
  item_node->ref = item;
  item_node->list = list;
  item_node->next = nullptr;
  if (list->tail)
  {
    item_node->prev = list->tail;
    list->tail->next = item_node;
    list->tail = item_node;
  }
  else
  {
    item_node->prev = nullptr;
    list->head = item_node;
    list->tail = item_node;
  }
}

ALWAYS_INLINE_RELEASE static void ListMoveToFront(GPUTextureCache::SourceList* list,
                                                  GPUTextureCache::SourceListNode* item_node)
{
  DebugAssert(list->head);
  if (!item_node->prev)
    return;

  item_node->prev->next = item_node->next;
  if (item_node->next)
    item_node->next->prev = item_node->prev;
  else
    list->tail = item_node->prev;

  item_node->prev = nullptr;
  list->head->prev = item_node;
  item_node->next = list->head;
  list->head = item_node;
}

ALWAYS_INLINE_RELEASE static void ListUnlink(const GPUTextureCache::SourceListNode& node)
{
  if (node.prev)
    node.prev->next = node.next;
  else
    node.list->head = node.next;
  if (node.next)
    node.next->prev = node.prev;
  else
    node.list->tail = node.prev;
}

template<typename F>
ALWAYS_INLINE_RELEASE static void ListIterate(const GPUTextureCache::SourceList& list, const F& f)
{
  for (const GPUTextureCache::Source::ListNode* n = list.head; n; n = n->next)
    f(n->ref);
}

[[maybe_unused]] ALWAYS_INLINE static TinyString SourceKeyToString(const GPUTextureCache::SourceKey& key)
{
  static constexpr const std::array<const char*, 4> texture_modes = {
    {"Palette4Bit", "Palette8Bit", "Direct16Bit", "Reserved_Direct16Bit"}};

  TinyString ret;
  if (key.mode < GPUTextureMode::Direct16Bit)
  {
    ret.format("{} Page[{}] CLUT@[{},{}]", texture_modes[static_cast<u8>(key.mode)], key.page, key.palette.GetXBase(),
               key.palette.GetYBase());
  }
  else
  {
    ret.format("{} Page[{}]", texture_modes[static_cast<u8>(key.mode)], key.page);
  }
  return ret;
}

[[maybe_unused]] ALWAYS_INLINE static TinyString SourceToString(const GPUTextureCache::Source* src)
{
  return SourceKeyToString(src->key);
}

ALWAYS_INLINE static u32 PageStartX(u32 pn)
{
  return (pn % GPUTextureCache::VRAM_PAGES_WIDE) * GPUTextureCache::VRAM_PAGE_WIDTH;
}

ALWAYS_INLINE static u32 PageStartY(u32 pn)
{
  return (pn / GPUTextureCache::VRAM_PAGES_WIDE) * GPUTextureCache::VRAM_PAGE_HEIGHT;
}

ALWAYS_INLINE_RELEASE static const u16* VRAMPagePointer(u32 pn)
{
  const u32 start_y = PageStartY(pn);
  const u32 start_x = PageStartX(pn);
  return &g_vram[start_y * VRAM_WIDTH + start_x];
}

// TODO: Vectorize these.
ALWAYS_INLINE_RELEASE static void DecodeTexture4(const u16* page, const u16* palette, u32* dest, u32 dest_stride)
{
  for (u32 y = 0; y < TEXTURE_PAGE_HEIGHT; y++)
  {
    const u16* page_ptr = page;
    u32* dest_ptr = dest;

    for (u32 x = 0; x < TEXTURE_PAGE_WIDTH / 4; x++)
    {
      const u32 pp = *(page_ptr++);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[pp & 0x0F]);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[(pp >> 4) & 0x0F]);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[(pp >> 8) & 0x0F]);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[pp >> 12]);
    }

    page += VRAM_WIDTH;
    dest = reinterpret_cast<u32*>(reinterpret_cast<u8*>(dest) + dest_stride);
  }
}
ALWAYS_INLINE_RELEASE static void DecodeTexture8(const u16* page, const u16* palette, u32* dest, u32 dest_stride)
{
  for (u32 y = 0; y < TEXTURE_PAGE_HEIGHT; y++)
  {
    const u16* page_ptr = page;
    u32* dest_ptr = dest;

    for (u32 x = 0; x < TEXTURE_PAGE_WIDTH / 2; x++)
    {
      const u32 pp = *(page_ptr++);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[pp & 0xFF]);
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(palette[pp >> 8]);
    }

    page += VRAM_WIDTH;
    dest = reinterpret_cast<u32*>(reinterpret_cast<u8*>(dest) + dest_stride);
  }
}
ALWAYS_INLINE_RELEASE static void DecodeTexture16(const u16* page, u32* dest, u32 dest_stride)
{
  for (u32 y = 0; y < TEXTURE_PAGE_HEIGHT; y++)
  {
    const u16* page_ptr = page;
    u32* dest_ptr = dest;

    for (u32 x = 0; x < TEXTURE_PAGE_WIDTH; x++)
      *(dest_ptr++) = VRAMRGBA5551ToRGBA8888(*(page_ptr++));

    page += VRAM_WIDTH;
    dest = reinterpret_cast<u32*>(reinterpret_cast<u8*>(dest) + dest_stride);
  }
}

ALWAYS_INLINE_RELEASE static void DecodeTexture(u8 page, GPUTexturePaletteReg palette, GPUTextureMode mode, u32* dest,
                                                u32 dest_stride)
{
  const u16* page_ptr = VRAMPagePointer(page);
  switch (mode)
  {
    case GPUTextureMode::Palette4Bit:
      DecodeTexture4(page_ptr, &g_vram[palette.GetYBase() * VRAM_WIDTH + palette.GetXBase()], dest, dest_stride);
      break;
    case GPUTextureMode::Palette8Bit:
      DecodeTexture8(page_ptr, &g_vram[palette.GetYBase() * VRAM_WIDTH + palette.GetXBase()], dest, dest_stride);
      break;
    case GPUTextureMode::Direct16Bit:
    case GPUTextureMode::Reserved_Direct16Bit:
      DecodeTexture16(page_ptr, dest, dest_stride);
      break;

      DefaultCaseIsUnreachable()
  }
}

static void DecodeTexture(u8 page, GPUTexturePaletteReg palette, GPUTextureMode mode, GPUTexture* texture)
{
  alignas(16) static u32 s_temp_buffer[TEXTURE_PAGE_WIDTH * TEXTURE_PAGE_HEIGHT];

  u32* tex_map;
  u32 tex_stride;
  const bool mapped =
    texture->Map(reinterpret_cast<void**>(&tex_map), &tex_stride, 0, 0, TEXTURE_PAGE_WIDTH, TEXTURE_PAGE_HEIGHT);
  if (!mapped)
  {
    tex_map = s_temp_buffer;
    tex_stride = sizeof(u32) * TEXTURE_PAGE_WIDTH;
  }

  DecodeTexture(page, palette, mode, tex_map, tex_stride);

  if (mapped)
    texture->Unmap();
  else
    texture->Update(0, 0, TEXTURE_PAGE_WIDTH, TEXTURE_PAGE_HEIGHT, tex_map, tex_stride);
}

ALWAYS_INLINE u32 WidthForMode(GPUTextureMode mode)
{
  return TEXTURE_PAGE_WIDTH >> ((mode < GPUTextureMode::Direct16Bit) ? (2 - static_cast<u8>(mode)) : 0);
}

GPUTextureCache::GPUTextureCache()
{
}

GPUTextureCache::~GPUTextureCache()
{
  Clear();
}

const GPUTextureCache::Source* GPUTextureCache::LookupSource(SourceKey key)
{
  GL_INS_FMT("TC: Lookup source {}", SourceKeyToString(key));

  SourceList& list = m_page_sources[key.page];
  for (SourceListNode* n = list.head; n; n = n->next)
  {
    if (n->ref->key == key)
    {
      GL_INS("TC: Source hit");
      ListMoveToFront(&list, n);
      return n->ref;
    }
  }

  return CreateSource(key);
}

template<typename F>
void GPUTextureCache::LoopPages(u32 x, u32 y, u32 width, u32 height, const F& f)
{
  DebugAssert(width > 0 && height > 0);
  DebugAssert((x + width) <= VRAM_WIDTH && (y + height) <= VRAM_HEIGHT);

  const u32 start_x = x / VRAM_PAGE_WIDTH;
  const u32 start_y = y / VRAM_PAGE_HEIGHT;
  const u32 end_x = (x + (width - 1)) / VRAM_PAGE_WIDTH;
  const u32 end_y = (y + (height - 1)) / VRAM_PAGE_HEIGHT;

  u32 page_number = PageIndex(start_x, start_y);
  for (u32 page_y = start_y; page_y <= end_y; page_y++)
  {
    u32 y_page_number = page_number;

    for (u32 page_x = start_x; page_x <= end_x; page_x++)
    {
      f(page_number);
      y_page_number++;
    }

    page_number += VRAM_PAGES_WIDE;
  }
}

void GPUTextureCache::Clear()
{
  for (u32 i = 0; i < NUM_PAGES; i++)
    InvalidatePage(i);

    // should all be null
#ifdef _DEBUG
  for (u32 i = 0; i < NUM_PAGES; i++)
    DebugAssert(!m_page_sources[i].head && !m_page_sources[i].tail);
#endif
}

void GPUTextureCache::InvalidatePage(u32 pn)
{
  DebugAssert(pn < NUM_PAGES);

  SourceList& ps = m_page_sources[pn];
  if (ps.head)
    GL_INS_FMT("Invalidate page {}", pn);

  for (SourceListNode* n = ps.head; n;)
  {
    Source* src = n->ref;
    n = n->next;

    GL_INS_FMT("Invalidate source {}", SourceToString(src));

    for (u32 i = 0; i < src->num_page_refs; i++)
      ListUnlink(src->page_refs[i]);

    if (src->from_hash_cache)
    {
      DebugAssert(src->from_hash_cache->ref_count > 0);
      src->from_hash_cache->ref_count--;
    }
    else
    {
      delete src->texture;
    }

    delete src;
  }

  ps.head = nullptr;
  ps.tail = nullptr;
}

void GPUTextureCache::InvalidatePages(u32 x, u32 y, u32 width, u32 height)
{
  LoopPages(x, y, width, height, [this](u32 page) { InvalidatePage(page); });
}

void GPUTextureCache::InvalidatePages(const Common::Rectangle<u32>& rc)
{
  InvalidatePages(rc.left, rc.top, rc.GetWidth(), rc.GetHeight());
}

const GPUTextureCache::Source* GPUTextureCache::CreateSource(SourceKey key)
{
  GL_INS_FMT("TC: Create source {}", SourceKeyToString(key));

  HashCacheEntry* hcentry = LookupHashCache(key);
  if (!hcentry)
  {
    GL_INS("TC: Hash cache lookup fail?!");
    return nullptr;
  }

  hcentry->ref_count++;
  hcentry->age = 0;

  Source* src = new Source();
  src->key = key;
  src->num_page_refs = 0;
  src->texture = hcentry->texture.get();
  src->from_hash_cache = hcentry;

  // Textures at front, CLUTs at back.
  std::array<u32, MAX_PAGE_REFS_PER_SOURCE> page_refns;
  const auto add_page_ref = [this, src, &page_refns](u32 pn) {
    // Don't double up references
    for (u32 i = 0; i < src->num_page_refs; i++)
    {
      if (page_refns[i] == pn)
        return;
    }

    const u32 ri = src->num_page_refs++;
    page_refns[ri] = pn;

    ListPrepend(&m_page_sources[pn], src, &src->page_refs[ri]);
  };
  const auto add_page_ref_back = [this, src, &page_refns](u32 pn) {
    // Don't double up references
    for (u32 i = 0; i < src->num_page_refs; i++)
    {
      if (page_refns[i] == pn)
        return;
    }

    const u32 ri = src->num_page_refs++;
    page_refns[ri] = pn;

    ListAppend(&m_page_sources[pn], src, &src->page_refs[ri]);
  };

  LoopPages(PageStartX(key.page), PageStartY(key.page), WidthForMode(key.mode), TEXTURE_PAGE_HEIGHT, add_page_ref);

  if (key.mode < GPUTextureMode::Direct16Bit)
  {
    LoopPages(key.palette.GetXBase(), key.palette.GetYBase(), GPUTexturePaletteReg::GetWidth(key.mode), 1,
              add_page_ref_back);
  }

  GL_INS_FMT("Appended new source {} to {} pages", SourceToString(src), src->num_page_refs);
  return src;
}

void GPUTextureCache::UpdateDrawnRect(const Common::Rectangle<u32>& rect)
{
  if (rect.left >= m_drawn_rect.left && rect.right <= m_drawn_rect.right && rect.top >= m_drawn_rect.top &&
      rect.bottom <= m_drawn_rect.bottom)
  {
    return;
  }

  m_drawn_rect.Include(rect);
}

void GPUTextureCache::InvalidateFromWrite(const Common::Rectangle<u32>& rect)
{
  if (m_drawn_rect.Intersects(rect))
  {
    // VRAM write includes area previously drawn to, we have to toss it all :(
    m_drawn_rect.Include(rect);
    GL_INS_FMT("TC: VRAM write {} intersects with draw area, draw area now {}", RectToString(rect),
               RectToString(m_drawn_rect));
    InvalidatePages(m_drawn_rect);
  }
  else
  {
    GL_INS_FMT("TC: Invalidate pages from VRAM write {}", RectToString(rect));
    InvalidatePages(rect);
  }
}

GPUTextureCache::HashType GPUTextureCache::HashPage(u8 page, GPUTextureMode mode) const
{
  XXH3_state_t state;
  XXH3_64bits_reset(&state);

  // Pages aren't contiguous in memory :(
  const u16* page_ptr = VRAMPagePointer(page);

  switch (mode)
  {
    case GPUTextureMode::Palette4Bit:
    {
      for (u32 y = 0; y < VRAM_PAGE_HEIGHT; y++)
      {
        XXH3_64bits_update(&state, page_ptr, VRAM_PAGE_WIDTH * sizeof(u16));
        page_ptr += VRAM_WIDTH;
      }
    }
    break;

    case GPUTextureMode::Palette8Bit:
    {
      for (u32 y = 0; y < VRAM_PAGE_HEIGHT; y++)
      {
        XXH3_64bits_update(&state, page_ptr, VRAM_PAGE_WIDTH * 2 * sizeof(u16));
        page_ptr += VRAM_WIDTH;
      }
    }
    break;

    case GPUTextureMode::Direct16Bit:
    {
      for (u32 y = 0; y < VRAM_PAGE_HEIGHT; y++)
      {
        XXH3_64bits_update(&state, page_ptr, VRAM_PAGE_WIDTH * 4 * sizeof(u16));
        page_ptr += VRAM_WIDTH;
      }
    }
    break;

      DefaultCaseIsUnreachable()
  }

  return XXH3_64bits_digest(&state);
}

GPUTextureCache::HashType GPUTextureCache::HashPalette(GPUTexturePaletteReg palette, GPUTextureMode mode) const
{
  const u16* base = &g_vram[palette.GetYBase() * VRAM_WIDTH + palette.GetXBase()];

  switch (mode)
  {
    case GPUTextureMode::Palette4Bit:
      return XXH3_64bits(base, sizeof(u16) * 16);

    case GPUTextureMode::Palette8Bit:
      return XXH3_64bits(base, sizeof(u16) * 256);

      DefaultCaseIsUnreachable()
  }
}

GPUTextureCache::HashCacheEntry* GPUTextureCache::LookupHashCache(SourceKey key)
{
  const HashType tex_hash = HashPage(key.page, key.mode);
  const HashType pal_hash = (key.mode < GPUTextureMode::Direct16Bit) ? HashPalette(key.palette, key.mode) : 0;
  const HashCacheKey hkey = {tex_hash, pal_hash, static_cast<HashType>(key.mode)};

  const auto it = m_hash_cache.find(hkey);
  if (it != m_hash_cache.end())
  {
    GL_INS_FMT("TC: Hash cache hit {:X} {:X}", hkey.texture_hash, hkey.palette_hash);
    return &it->second;
  }

  GL_INS_FMT("TC: Hash cache miss {:X} {:X}", hkey.texture_hash, hkey.palette_hash);

  HashCacheEntry entry;
  entry.ref_count = 0;
  entry.age = 0;
  entry.texture = g_gpu_device->FetchTexture(TEXTURE_PAGE_WIDTH, TEXTURE_PAGE_HEIGHT, 1, 1, 1,
                                             GPUTexture::Type::Texture, GPUTexture::Format::RGBA8);
  if (!entry.texture)
  {
    Log_ErrorPrint("Failed to create texture.");
    return nullptr;
  }

  DecodeTexture(key.page, key.palette, key.mode, entry.texture.get());

  return &m_hash_cache.emplace(hkey, std::move(entry)).first->second;
}

void GPUTextureCache::RemoveFromHashCache(HashCache::iterator it)
{
  g_gpu_device->RecycleTexture(std::move(it->second.texture));
  m_hash_cache.erase(it);
}

void GPUTextureCache::AgeHashCache()
{
  // Number of frames before unused hash cache entries are evicted.
  static constexpr u32 MAX_HASH_CACHE_AGE = 600;

  // Maximum number of textures which are permitted in the hash cache at the end of the frame.
  static constexpr u32 MAX_HASH_CACHE_SIZE = 200;

  bool might_need_cache_purge = (m_hash_cache.size() > MAX_HASH_CACHE_SIZE);
  if (might_need_cache_purge)
    s_hash_cache_purge_list.clear();

  for (auto it = m_hash_cache.begin(); it != m_hash_cache.end();)
  {
    HashCacheEntry& e = it->second;
    if (e.ref_count > 0)
    {
      ++it;
      continue;
    }

    if (++e.age > MAX_HASH_CACHE_AGE)
    {
      RemoveFromHashCache(it++);
      continue;
    }

    // We might free up enough just with "normal" removals above.
    if (might_need_cache_purge)
    {
      might_need_cache_purge = (m_hash_cache.size() > MAX_HASH_CACHE_SIZE);
      if (might_need_cache_purge)
        s_hash_cache_purge_list.emplace_back(it, static_cast<s32>(e.age));
    }

    ++it;
  }

  // Pushing to a list, sorting, and removing ends up faster than re-iterating the map.
  if (might_need_cache_purge)
  {
    std::sort(s_hash_cache_purge_list.begin(), s_hash_cache_purge_list.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    const u32 entries_to_purge = std::min(static_cast<u32>(m_hash_cache.size() - MAX_HASH_CACHE_SIZE),
                                          static_cast<u32>(s_hash_cache_purge_list.size()));
    for (u32 i = 0; i < entries_to_purge; i++)
      RemoveFromHashCache(s_hash_cache_purge_list[i].first);
  }
}

size_t GPUTextureCache::HashCacheKeyHash::operator()(const HashCacheKey& k) const
{
  std::size_t h = 0;
  hash_combine(h, k.texture_hash, k.palette_hash, k.mode);
  return h;
}
