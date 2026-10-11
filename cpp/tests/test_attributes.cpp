#include <gtest/gtest.h>

#include "minivec/hnsw.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{
    using minivec::HNSWIndexSimple;

    bool bitmap_has(const HNSWIndexSimple::BitmapResult &bitmap, std::size_t id)
    {
        return (bitmap.words[id / 64] & (std::uint64_t{1} << (id % 64))) != 0;
    }

    TEST(HNSWAttributeStore, PredicatesReturnAlignedBitmapAndSelectivity)
    {
        HNSWIndexSimple index(2, 4, 20, 10, true, 7);
        std::vector<float> vector(2);
        for (int id = 0; id < 70; ++id)
        {
            vector[0] = static_cast<float>(id);
            vector[1] = 0.0F;
            std::unordered_map<std::string, std::int64_t> attrs;
            if (id % 3 != 0)
                attrs["group"] = id % 5;
            if (id % 2 == 0)
                attrs["year"] = 2020 + id;
            EXPECT_EQ(index.insert_vector(vector.data(), attrs), id);
        }

        const auto equal = index.attribute_eq("group", 2);
        EXPECT_EQ(equal.words.size(), 2U);
        EXPECT_TRUE(bitmap_has(equal, 2));
        EXPECT_TRUE(bitmap_has(equal, 7));
        EXPECT_FALSE(bitmap_has(equal, 0));
        EXPECT_EQ(equal.count, 10);

        const auto membership = index.attribute_in("group", {1, 2, 2});
        EXPECT_EQ(membership.count, 19);
        EXPECT_TRUE(bitmap_has(membership, 1));
        EXPECT_TRUE(bitmap_has(membership, 2));
        EXPECT_FALSE(bitmap_has(membership, 3));

        const auto range = index.attribute_range("year", 2040, 2050);
        EXPECT_EQ(range.count, 6);
        EXPECT_TRUE(bitmap_has(range, 20));
        EXPECT_TRUE(bitmap_has(range, 30));
        EXPECT_FALSE(bitmap_has(range, 21));
        EXPECT_EQ(index.attribute_eq("absent", 1).count, 0);
    }

    TEST(HNSWAttributeStore, SetAndClearKeepColumnsAligned)
    {
        HNSWIndexSimple index(2, 4, 20, 10, true, 9);
        const float first[] = {0.0F, 0.0F};
        const float second[] = {1.0F, 1.0F};
        const auto first_id = index.insert_vector(first);
        index.set_attribute(first_id, "tenant", -1);
        const auto second_id = index.insert_vector(second, {{"tenant", 4}});
        EXPECT_EQ(index.attribute_in("tenant", {-1, 4}).count, 2);
        EXPECT_EQ(index.attribute_eq("tenant", 4).count, 1);

        index.clear();
        EXPECT_TRUE(index.attribute_eq("tenant", 4).words.empty());
        EXPECT_EQ(index.get_node_count(), 0);
    }

    TEST(HNSWAttributeStore, RejectsInvalidRange)
    {
        HNSWIndexSimple index(2, 4, 20, 10);
        EXPECT_THROW(index.attribute_range("year", 5, 4), std::invalid_argument);
    }
}
